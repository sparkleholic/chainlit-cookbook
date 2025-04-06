import json
import chainlit as cl
import httpx

from logging_transport import LoggingTransport

SYSTEM = "you are a helpful assistant."
MODEL_NAME = "EXAONE-3.5-2.4B-Instruct"

transport = LoggingTransport(httpx.AsyncHTTPTransport())
client = httpx.AsyncClient(base_url="http://localhost:8080", transport=transport)


@cl.step(type="tool")
async def move_map_to(latitude: float, longitude: float):
    await open_map()

    fn = cl.CopilotFunction(
        name="move-map", args={"latitude": latitude, "longitude": longitude}
    )
    await fn.acall()

    return "Map moved!"


tools = [
    {
      "type": "function",
      "function": {
        "name": "move_map_to",
        "description": "Move the map to the given latitude and longitude.",
        "parameters": {
          "type": "object",
          "properties": {
            "latitude": {
              "type": "string",
              "description": "The latitude of the location to move the map to"
            },
            "longitude": {
              "type": "string",
              "description": "The longitude of the location to move the map to"
            }
          },
          "required": ["latitude", "longitude"]
        }
      }
    }
]

TOOL_FUNCTIONS = {
    "move_map_to": move_map_to,
}


async def call_llama_server(chat_messages):
    msg = cl.Message(content="", author="Llama")

    payload = {
        "system": SYSTEM,
        "model": MODEL_NAME,
        "messages": chat_messages,
        "tools": tools,
        "max_tokens": 1024,
    }

    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        async for line in response.aiter_lines():
            if line.strip():
                data = json.loads(line)
                if "text" in data:
                    await msg.stream_token(data["text"])

    await msg.send()
    final_response = response.json()  # Removed 'await' here

    return final_response


async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input

    tool_function = TOOL_FUNCTIONS.get(tool_name)

    if tool_function:
        try:
            return await tool_function(**tool_input)
        except TypeError:
            return json.dumps({"error": f"Invalid input for {tool_name}"})
    else:
        return json.dumps({"error": f"Invalid tool: {tool_name}"})


async def open_map():
    map_props = {"latitude": 37.7749, "longitude": -122.4194, "zoom": 12}
    custom_element = cl.CustomElement(name="Map", props=map_props, display="inline")
    await cl.ElementSidebar.set_title("canvas")
    await cl.ElementSidebar.set_elements([custom_element], key="map-canvas")


@cl.action_callback("close_map")
async def on_test_action():
    await cl.ElementSidebar.set_elements([])


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Paris",
            message="Show me Paris.",
        ),
        cl.Starter(
            label="NYC",
            message="Show me NYC.",
        ),
        cl.Starter(
            label="Tokyo",
            message="Show me Tokyo.",
        ),
    ]


@cl.on_chat_start
async def on_start():
    cl.user_session.set("chat_messages", [])

    await open_map()


@cl.on_message
async def on_message(msg: cl.Message):
    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "user", "content": msg.content})
    response = await call_llama_server(chat_messages)

    # Validate response structure
    if "choices" not in response or not response["choices"]:
        raise ValueError("Invalid response: 'choices' key is missing or empty.")

    while "tool_calls" in response["choices"][0]["message"]:
        tool_call = response["choices"][0]["message"]["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])

        tool_function = TOOL_FUNCTIONS.get(function_name)
        if tool_function:
            tool_result = await tool_function(**function_args)
        else:
            tool_result = {"error": f"Function '{function_name}' not found."}

        messages = [
            {"role": "assistant", "content": response["choices"][0]["message"]["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": str(tool_result),
                    }
                ],
            },
        ]

        chat_messages.extend(messages)
        response = await call_llama_server(chat_messages)

    final_response = response["choices"][0]["message"]["content"]

    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "assistant", "content": final_response})
