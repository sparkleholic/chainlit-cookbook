import httpx

class LoggingTransport(httpx.AsyncBaseTransport):
    def __init__(self, wrapped_transport):
        self.wrapped_transport = wrapped_transport

    async def handle_async_request(self, request):
        print("🔽 HTTP REQUEST")
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print("Headers:")
        for k, v in request.headers.items():
            print(f"  {k}: {v}")
        if request.content:
            try:
                body = request.content.decode()
            except AttributeError:
                body = request.content
            print("Body:")
            print(body)

        response = await self.wrapped_transport.handle_async_request(request)

        print("🔼 HTTP RESPONSE")
        print(f"Status Code: {response.status_code}")
        print("Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")

        content = await response.aread()
        print("Body:")
        print(content.decode())

        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=content,
            request=request,
            extensions=response.extensions,
        )