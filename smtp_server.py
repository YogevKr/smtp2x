import newrelic.agent
newrelic.agent.initialize()

import asyncio
import logging
from email import message_from_bytes
from email.policy import default
import base64
import os
import aiohttp
from io import BytesIO
from dataclasses import dataclass
from openai import AsyncOpenAI
import json
from pydantic import BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmailMessage:
    sender: str
    recipient: str
    subject: str
    body: str
    image_data: bytes = None

class AnalyticEvent(BaseModel):
    analysis: str
    result: bool

class SMTPConfig:
    def __init__(self, hostname, max_message_size, client_timeout, openai_api_key, telegram_bot_token, telegram_chat_id, gpt4_task, pagerduty_trigger):
        self.hostname = hostname
        self.max_message_size = max_message_size
        self.client_timeout = client_timeout
        self.openai_api_key = openai_api_key
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.gpt4_task = gpt4_task
        self.pagerduty_trigger = pagerduty_trigger

class MessageTransformer:
    @staticmethod
    async def transform(email_data):
        email_message = message_from_bytes(email_data, policy=default)
        
        image_data = None
        for part in email_message.walk():
            if part.get_content_maintype() == 'image':
                image_data = part.get_payload(decode=True)
                break
        
        return EmailMessage(
            sender=email_message['From'],
            recipient=email_message['To'],
            subject=email_message['Subject'],
            body=email_message.get_body(preferencelist=('plain', 'html')).get_content(),
            image_data=image_data
        )

class GPT4Analyzer:
    def __init__(self, api_key, task):
        self.client = AsyncOpenAI(api_key=api_key)
        self.task = task

    async def analyze(self, image_data):
        logger.info("Uploading image to OpenAI GPT-4")
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        try:
            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.task},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format=AnalyticEvent
            )
            
            event = response.choices[0].message.parsed
            logger.info(f"GPT-4 response: {response}")
            return event.result
        except Exception as e:
            logger.error(f"Error from OpenAI API: {str(e)}")
            return False

class TelegramSender:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send(self, image_data):
        logger.info("Sending image to Telegram")
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        image_file = BytesIO(image_data)
        image_file.name = 'image.jpg'
        
        form = aiohttp.FormData()
        form.add_field('chat_id', self.chat_id)
        form.add_field('photo', image_file)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    logger.info("Image sent to Telegram successfully")
                else:
                    logger.error(f"Error sending image to Telegram: {response.status}")

class PagerDutyTrigger:
    def __init__(self, api_token, service_id, from_email):
        self.api_token = api_token
        self.service_id = service_id
        self.from_email = from_email
        self.url = "https://api.pagerduty.com/incidents"

    async def trigger_incident(self, title, details, urgency="high"):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/vnd.pagerduty+json;version=2",
            "Authorization": f"Token token={self.api_token}",
            "From": self.from_email
        }
        
        payload = {
            "incident": {
                "type": "incident",
                "title": title,
                "service": {
                    "id": self.service_id,
                    "type": "service_reference"
                },
                "urgency": urgency,
                "body": {
                    "type": "incident_body",
                    "details": details
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    if response.status == 201:
                        logger.info("PagerDuty incident created successfully")
                        return json.loads(response_text)
                    else:
                        logger.error(f"Failed to create PagerDuty incident. Status code: {response.status}")
                        logger.error(f"Response: {response_text}")
                        logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
                        logger.error(f"Request headers: {json.dumps(headers, indent=2)}")
                        try:
                            error_data = json.loads(response_text)
                            if 'error' in error_data:
                                logger.error(f"Error message: {error_data['error'].get('message')}")
                                logger.error(f"Error code: {error_data['error'].get('code')}")
                        except json.JSONDecodeError:
                            logger.error("Could not parse error response as JSON")
                        return None
            except aiohttp.ClientError as e:
                logger.error(f"Network error when creating PagerDuty incident: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error when creating PagerDuty incident: {str(e)}")
        return None

class ProcessingPipeline:
    def __init__(self, config):
        self.transformer = MessageTransformer()
        self.analyzer = GPT4Analyzer(config.openai_api_key, config.gpt4_task)
        self.sender = TelegramSender(config.telegram_bot_token, config.telegram_chat_id)
        self.pagerduty = config.pagerduty_trigger

    async def process(self, email_data):
        email_message = await self.transformer.transform(email_data)
        logger.info(f"Processed email: From: {email_message.sender}, Subject: {email_message.subject}")

        if email_message.image_data:
            analysis_result = await self.analyzer.analyze(email_message.image_data)

            if analysis_result:
                await self.sender.send(email_message.image_data)
                incident = await self.pagerduty.trigger_incident(
                    title=f"Critical image detected: {email_message.subject}",
                    details=f"A critical image was detected in an email from {email_message.sender}. Subject: {email_message.subject}",
                    urgency="high"
                )
                if incident:
                    logger.info(f"PagerDuty incident created: {incident.get('incident', {}).get('id')}")
                else:
                    logger.error("Failed to create PagerDuty incident")
            else:
                logger.info("Analysis result is negative. Not sending to Telegram or triggering PagerDuty.")
        else:
            logger.info("No image attachment found in the email.")

class SMTPServer:
    def __init__(self, config):
        self.config = config
        self.pipeline = ProcessingPipeline(config)

    async def start(self, host, port):
        server = await asyncio.start_server(self.handle_client, host, port)
        addr = server.sockets[0].getsockname()
        logger.info(f'SMTP server listening on {addr}')

        async with server:
            await server.serve_forever()

    async def handle_client(self, reader, writer):
        session = SMTPSession(self, reader, writer, self.config)
        await session.handle()

class SMTPSession:
    def __init__(self, server, reader, writer, config):
        self.server = server
        self.reader = reader
        self.writer = writer
        self.config = config
        self.remote_addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {self.remote_addr}")

    async def handle(self):
        try:
            await self.write_response(f"220 {self.config.hostname} SMTP Server")

            while True:
                line = await self.read_line()
                if not line:
                    logger.warning(f"Client {self.remote_addr} disconnected")
                    break

                logger.debug(f"Received command: {line}")
                parts = line.split(None, 1)
                command = parts[0].upper() if parts else ''

                handler = getattr(self, f"handle_{command.lower()}", None)
                if handler:
                    await handler(line)
                else:
                    logger.warning(f"Unknown command received: {command}")
                    await self.write_response("250 OK")  # Be more permissive

                if command == "QUIT":
                    break

        except Exception as e:
            logger.error(f"Error handling client {self.remote_addr}: {e}", exc_info=True)
        finally:
            logger.info(f"Closing connection from {self.remote_addr}")
            self.writer.close()
            await self.writer.wait_closed()

    async def read_line(self):
        try:
            line = await asyncio.wait_for(self.reader.readline(), timeout=self.config.client_timeout)
            return line.decode().strip()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout reading from client {self.remote_addr}")
            return None

    async def write_response(self, response):
        logger.debug(f"Sending response: {response}")
        self.writer.write(f"{response}\r\n".encode())
        await self.writer.drain()

    async def handle_ehlo(self, line):
        logger.info(f"EHLO command received: {line}")
        await self.write_response(f"250-{self.config.hostname}")
        await self.write_response(f"250-SIZE {self.config.max_message_size}")
        await self.write_response("250-PIPELINING")
        await self.write_response("250-ENHANCEDSTATUSCODES")
        await self.write_response("250-AUTH LOGIN PLAIN")
        await self.write_response("250 HELP")

    async def handle_helo(self, line):
        logger.info(f"HELO command received: {line}")
        await self.write_response(f"250 Hello {line.split()[1]}")

    async def handle_mail(self, line):
        logger.info(f"MAIL FROM command received: {line}")
        await self.write_response("250 OK")

    async def handle_rcpt(self, line):
        logger.info(f"RCPT TO command received: {line}")
        await self.write_response("250 OK")

    async def handle_data(self, line):
        logger.info("DATA command received")
        await self.write_response("354 End data with <CR><LF>.<CR><LF>")

        email_data = b""
        while True:
            line = await self.reader.readline()
            if line.strip() == b".":
                break
            email_data += line

        await self.server.pipeline.process(email_data)
        await self.write_response("250 OK")

    async def handle_quit(self, line):
        logger.info("QUIT command received")
        await self.write_response("221 Bye")

    async def handle_auth(self, line):
        logger.info(f"AUTH command received: {line}")
        # For now, accept any credentials
        await self.write_response("235 Authentication successful")

async def main():
    pagerduty_token = os.getenv("PAGERDUTY_API_TOKEN")
    pagerduty_service_id = os.getenv("PAGERDUTY_SERVICE_ID")
    pagerduty_from_email = os.getenv("PAGERDUTY_FROM_EMAIL")
    
    config = SMTPConfig(
        hostname="192.168.40.191",
        max_message_size=50 * 1024 * 1024,  # 50 MB
        client_timeout=300,  # 5 minutes
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        gpt4_task=os.getenv("GPT4_TASK"),
        pagerduty_trigger=PagerDutyTrigger(pagerduty_token, pagerduty_service_id, pagerduty_from_email)
    )

    server = SMTPServer(config)
    await server.start('0.0.0.0', 2525)

if __name__ == "__main__":
    asyncio.run(main())
