import os
import json
import requests
from PIL import Image
import io
import plugins
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from channel.chat_message import ChatMessage
from common.log import logger
from plugins import *
from gradio_client import Client

@plugins.register(
    name="image_processor",
    desire_priority=0,
    hidden=False,
    desc="A plugin that downloads and processes images using Gradio API",
    version="1.0",
    author="your_name",
)
class ImageProcessor(Plugin):
    def init(self):
        super().init()
        self.client = Client("openbmb/MiniCPM-Llama3-V-2_5")
        self.config = self.load_config()
        logger.info("[image_processor] Initialized")

    def load_config(self):
        curdir = os.path.dirname(__file__)
        config_path = os.path.join(curdir, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return {}

    def on_handle_context(self, e_context: EventContext):
        if e_context["context"].type not in [ContextType.TEXT, ContextType.IMAGE]:
            return

        content = e_context['context'].content
        if e_context["context"].type == ContextType.TEXT:
            # Handle text command to trigger image processing
            if "process image" in content.lower():
                self.process_image(e_context)
            e_context.action = EventAction.BREAK
            return

    def process_image(self, e_context):
        image_url = self.config.get("image_url")
        image_path = self.download_image(image_url)
        if not image_path:
            e_context["context"].content = "Failed to download image"
            return

        # Upload the image and get the description
        upload_result = self.client.predict(
            image=image_path,
            _chatbot=[],
            api_name="/upload_img"
        )
        logger.info(f"Image upload result: {upload_result}")

        # Now send a text question to describe the image
        question_result = self.client.predict(
            _question="请详细且充分的描述图像内容及相关信息",
            _chat_bot=[],
            params_form="Sampling",
            num_beams=3,
            repetition_penalty=1.2,
            repetition_penalty_2=1.05,
            top_p=0.8,
            top_k=100,
            temperature=0.7,
            api_name="/respond"
        )
        logger.info(f"Question result: {question_result}")
        e_context["context"].content = f"Image description: {question_result}"

    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image_path = os.path.join("temp_images", os.path.basename(url))
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return image_path
        else:
            logger.error("Failed to download image")
            return None
