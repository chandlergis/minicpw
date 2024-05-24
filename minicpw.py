import requests
from gradio_client import Client
import plugins
from plugins import *
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger

# 初始化Gradio客户端
client = Client("openbmb/MiniCPM-Llama3-V-2_5")

@plugins.register(name="ImageRecognitionPlugin",
                  desc="识别图片内容",
                  version="1.0",
                  author="Cool",
                  desire_priority=100)
class ImageRecognitionPlugin(Plugin):

    def __init__(self):
        super().__init__()
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        logger.info(f"[{__class__.__name__}] initialized")

    def get_help_text(self, **kwargs):
        return "发送图片获取内容描述"

    def on_handle_context(self, e_context: EventContext):
        if e_context['context'].type != ContextType.IMAGE:
            return
        image_url = e_context['context'].content
        image_path = self.download_image(image_url)
        if image_path:
            description = self.recognize_image(image_path)
            if description:
                reply = Reply(type=ReplyType.TEXT, content=description)
            else:
                reply = Reply(type=ReplyType.ERROR, content="无法识别图片内容，请稍后再试。")
        else:
            reply = Reply(type=ReplyType.ERROR, content="图片下载失败。")
        
        e_context["reply"] = reply
        e_context.action = EventAction.BREAK_PASS

    def download_image(self, image_url):
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_path = 'downloaded_image.png'
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                return image_path
            else:
                logger.error("Failed to download image")
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
        return None

    def recognize_image(self, image_path):
        try:
            upload_result = client.predict(
                image=image_path,
                _chatbot=[],
                api_name="/upload_img"
            )
            if upload_result:
                # Assuming the response includes a description
                return upload_result.get('description', 'No description found')
        except Exception as e:
            logger.error(f"Error recognizing image: {e}")
        return "识别失败"

