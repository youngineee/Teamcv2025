import asyncio, json, cv2
from pathlib import Path
from aiohttp import web
from aiortc import (
    RTCPeerConnection, RTCSessionDescription,
    RTCConfiguration, RTCIceServer, RTCIceCandidate
)

pcs = set()

async def index(request):
    base = Path(__file__).resolve().parent
    html = base / "index_ws.html"
    if html.exists():
        return web.FileResponse(html)
    else:
        return web.Response(text="OK (HTTP up, but index_ws.html not found)")

async def health(request):
    return web.Response(text="ok")

async def wait_ice_complete(pc: RTCPeerConnection):
    if pc.iceGatheringState == "complete":
        return
    fut = asyncio.get_event_loop().create_future()
    @pc.on("icegatheringstatechange")
    def _on_ice():
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(True)
    await fut

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    config = RTCConfiguration(iceServers=[RTCIceServer('stun:stun.l.google.com:19302')])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)
    print("PeerConnection created")

    @pc.on("iceconnectionstatechange")
    def _ice_state():
        print("ICE state:", pc.iceConnectionState)

    @pc.on("connectionstatechange")
    def _pc_state():
        print("PC state:", pc.connectionState)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            cv2.namedWindow("Phone Camera (WS signaling)", cv2.WINDOW_NORMAL)
            async def consume():
                try:
                    while True:
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        cv2.imshow("Phone Camera (WS signaling)", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    cv2.destroyAllWindows()
            asyncio.create_task(consume())

    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                break
            data = json.loads(msg.data)

            if data["type"] == "offer":
                offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await wait_ice_complete(pc)
                await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})

            elif data["type"] == "candidate":
                cand = data.get("candidate")
                try:
                    if cand is None:
                        await pc.addIceCandidate(None)
                    else:
                        await pc.addIceCandidate(RTCIceCandidate(
                            sdpMid=cand.get("sdpMid"),
                            sdpMLineIndex=cand.get("sdpMLineIndex"),
                            candidate=cand.get("candidate"),
                        ))
                except Exception as e:
                    print("addIceCandidate error:", e)

            elif data["type"] == "bye":
                break

    finally:
        await pc.close()
        pcs.discard(pc)
        try:
            await ws.close()
        except:
            pass
        cv2.destroyAllWindows()
        print("PeerConnection closed")

    return ws

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()
    cv2.destroyAllWindows()

app = web.Application()
app.router.add_get("/", index)
app.router.add_get("/health", health)
app.router.add_get("/ws", ws_handler)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    print("HTTP server starting on http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)   # 🔓 HTTPS → HTTP
