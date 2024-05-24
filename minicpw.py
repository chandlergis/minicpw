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

    def on_handle_context(self, e_context: EventContext):
        if e_context["context"].type == ContextType.IMAGE:
            image_path = e_context['context'].content  # Assuming the content contains the image path or URL
            self.process_image(image_path, e_context)
        elif e_context["context"].type == ContextType.TEXT and "process image" in e_context['context'].content.lower():
            # Use default image URL from config if specific command is given in text
            image_path = self.config.get("image_url")
            self.process_image(image_path, e_context)
        e_context.action = EventAction.BREAK

    def process_image(self, image_url, e_context):
        image_path = self.download_image(image_url)
        if not image_path:
            e_context["context"].content = "Failed to download image"
            return

        # Upload the image and get the description
        upload_result = self.client.predict(
            image=image_path,
            _chatbot=[],
            api_name=self.config['api_names']['upload_image']
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
            api_name=self.config['api_names']['respond']
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
