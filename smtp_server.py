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
import time
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from typing import Any

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

class PagerDutyTrigger:
    def __init__(self, api_token, service_id, from_email):
        self.api_token = api_token
        self.service_id = service_id
        self.from_email = from_email
        self.url = "https://api.pagerduty.com/incidents"

    @newrelic.agent.function_trace()
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
                with newrelic.agent.FunctionTrace(name='pagerduty_api_call', group='External'):
                    async with session.post(self.url, json=payload, headers=headers) as response:
                        response_text = await response.text()
                        if response.status == 201:
                            logger.info("PagerDuty incident created successfully")
                            newrelic.agent.record_custom_event('PagerDutyTrigger', {'status': 'success'})
                            return json.loads(response_text)
                        else:
                            logger.error(f"Failed to create PagerDuty incident. Status code: {response.status}")
                            logger.error(f"Response: {response_text}")
                            newrelic.agent.record_custom_event('PagerDutyTrigger', {'status': 'error', 'error_code': response.status})
                            return None
            except aiohttp.ClientError as e:
                logger.error(f"Network error when creating PagerDuty incident: {str(e)}")
                newrelic.agent.record_exception()
            except Exception as e:
                logger.error(f"Unexpected error when creating PagerDuty incident: {str(e)}")
                newrelic.agent.record_exception()
        return None

class SMTPConfig:
    def __init__(self, hostname: str, max_message_size: int, client_timeout: int, 
                 enabled_analyzers: list[str],
                 openai_api_key: str | None = None,
                 gemini_api_key: str | None = None,
                 telegram_bot_token: str | None = None,
                 telegram_chat_id: str | None = None,
                 gpt4_task: str | None = None,
                 gemini_task: str | None = None,
                 pagerduty_trigger: PagerDutyTrigger | None = None):
        self.hostname = hostname
        self.max_message_size = max_message_size
        self.client_timeout = client_timeout
        self.enabled_analyzers = enabled_analyzers
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.gpt4_task = gpt4_task
        self.gemini_task = gemini_task
        self.pagerduty_trigger = pagerduty_trigger

class MessageTransformer:
    @staticmethod
    @newrelic.agent.function_trace()
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

    @newrelic.agent.function_trace()
    async def analyze(self, image_data):
        logger.info("Uploading image to OpenAI GPT-4")
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        try:
            with newrelic.agent.FunctionTrace(name='gpt4_api_call', group='External'):
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
            newrelic.agent.record_custom_event('GPT4Analysis', {'result': event.result})
            return event.result
        except Exception as e:
            logger.error(f"Error from OpenAI API: {str(e)}")
            newrelic.agent.record_exception()
            return False

class GeminiAnalyzer:
    def __init__(self, api_key: str, task: str):
        genai.configure(api_key=api_key)
        self.task = task
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                required=["analysis", "result"],
                properties={
                    "analysis": content.Schema(type=content.Type.STRING),
                    "result": content.Schema(type=content.Type.BOOLEAN),
                },
            ),
            "response_mime_type": "application/json"
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-002",
            generation_config=self.generation_config
        )

    @newrelic.agent.function_trace()
    async def analyze(self, image_data: bytes) -> bool:
        logger.info("Analyzing image with Gemini")
        
        try:
            with newrelic.agent.FunctionTrace(name='gemini_api_call', group='External'):
                # Save image temporarily
                temp_path = "/tmp/temp_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                
                # Upload to Gemini
                image_file = genai.upload_file(temp_path, mime_type="image/jpeg")
                
                # Create chat session
                chat = self.model.start_chat()
                response = chat.send_message([image_file, self.task])
                
                # Parse response
                result = json.loads(response.text)
                logger.info(f"Gemini response: {result}")
                newrelic.agent.record_custom_event('GeminiAnalysis', {
                    'status': 'success',
                    'result': result['result']
                })
                
                # Cleanup
                os.remove(temp_path)
                
                return bool(result['result'])
                
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Error from Gemini API: {error_msg}")
            
            # Record detailed error information to New Relic
            newrelic.agent.record_custom_event('GeminiAnalysis', {
                'status': 'error',
                'error_type': error_type,
                'error_message': error_msg
            })
            newrelic.agent.record_exception(params={
                'error_type': error_type,
                'analyzer': 'gemini'
            })
            
            # Re-raise the exception to be handled by the calling code
            raise

class TelegramSender:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    @newrelic.agent.function_trace()
    async def send(self, image_data):
        logger.info("Sending image to Telegram")
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        image_file = BytesIO(image_data)
        image_file.name = 'image.jpg'
        
        form = aiohttp.FormData()
        form.add_field('chat_id', self.chat_id)
        form.add_field('photo', image_file)
        
        async with aiohttp.ClientSession() as session:
            with newrelic.agent.FunctionTrace(name='telegram_api_call', group='External'):
                async with session.post(url, data=form) as response:
                    if response.status == 200:
                        logger.info("Image sent to Telegram successfully")
                        newrelic.agent.record_custom_event('TelegramSend', {'status': 'success'})
                    else:
                        logger.error(f"Error sending image to Telegram: {response.status}")
                        newrelic.agent.record_custom_event('TelegramSend', {'status': 'error', 'error_code': response.status})

class ProcessingPipeline:
    def __init__(self, config: SMTPConfig):
        self.transformer = MessageTransformer()
        self.analyzers: list[tuple[str, Any]] = []
        
        if "gpt4" in config.enabled_analyzers and config.openai_api_key:
            self.analyzers.append(("gpt4", GPT4Analyzer(config.openai_api_key, config.gpt4_task)))
            
        if "gemini" in config.enabled_analyzers and config.gemini_api_key:
            self.analyzers.append(("gemini", GeminiAnalyzer(config.gemini_api_key, config.gemini_task)))
            
        if not self.analyzers:
            raise ValueError("No analyzers configured! Set ENABLED_ANALYZERS env var.")
            
        self.sender = TelegramSender(config.telegram_bot_token, config.telegram_chat_id)
        self.pagerduty = config.pagerduty_trigger

    @newrelic.agent.function_trace()
    async def process(self, email_data: bytes) -> None:
        email_message = await self.transformer.transform(email_data)
        logger.info(f"Processed email: From: {email_message.sender}, Subject: {email_message.subject}")

        if email_message.image_data:
            # Run all enabled analyzers in parallel
            analysis_tasks = [
                analyzer.analyze(email_message.image_data) 
                for name, analyzer in self.analyzers
            ]
            results = await asyncio.gather(*analysis_tasks)
            
            # Log results from each analyzer
            for (name, _), result in zip(self.analyzers, results):
                logger.info(f"{name} analysis result: {result}")
            
            # Trigger actions if any analyzer returns True
            if any(results):
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
                logger.info("All analysis results are negative. Not sending to Telegram or triggering PagerDuty.")
        else:
            logger.info("No image attachment found in the email.")

class SMTPServer:
    def __init__(self, config):
        self.config = config
        self.pipeline = ProcessingPipeline(config)
        self.is_running = False
        self.server = None

    @newrelic.agent.background_task(name='smtp2x_start_server')
    async def start(self, host, port):
        self.is_running = True
        self.server = await asyncio.start_server(self.handle_client, host, port)
        addr = self.server.sockets[0].getsockname()
        logger.info(f'smtp2x service listening on {addr}')

        async with self.server:
            await self.server.serve_forever()

    def stop(self):
        logger.info("Stopping smtp2x service")
        self.is_running = False
        if self.server:
            self.server.close()

    @newrelic.agent.function_trace()
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

    @newrelic.agent.background_task()
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
            newrelic.agent.record_exception()
        finally:
            logger.info(f"Closing connection from {self.remote_addr}")
            self.writer.close()
            await self.writer.wait_closed()

    @newrelic.agent.function_trace()
    async def read_line(self):
        try:
            line = await asyncio.wait_for(self.reader.readline(), timeout=self.config.client_timeout)
            return line.decode().strip()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout reading from client {self.remote_addr}")
            newrelic.agent.record_custom_event('SMTPTimeout', {'remote_addr': str(self.remote_addr)})
            return None

    @newrelic.agent.function_trace()
    async def write_response(self, response):
        logger.debug(f"Sending response: {response}")
        self.writer.write(f"{response}\r\n".encode())
        await self.writer.drain()

    @newrelic.agent.function_trace()
    async def handle_ehlo(self, line):
        logger.info(f"EHLO command received: {line}")
        await self.write_response(f"250-{self.config.hostname}")
        await self.write_response(f"250-SIZE {self.config.max_message_size}")
        await self.write_response("250-PIPELINING")
        await self.write_response("250-ENHANCEDSTATUSCODES")
        await self.write_response("250-AUTH LOGIN PLAIN")
        await self.write_response("250 HELP")

    @newrelic.agent.function_trace()
    async def handle_helo(self, line):
        logger.info(f"HELO command received: {line}")
        await self.write_response(f"250 Hello {line.split()[1]}")

    @newrelic.agent.function_trace()
    async def handle_mail(self, line):
        logger.info(f"MAIL FROM command received: {line}")
        await self.write_response("250 OK")

    @newrelic.agent.function_trace()
    async def handle_rcpt(self, line):
        logger.info(f"RCPT TO command received: {line}")
        await self.write_response("250 OK")

    @newrelic.agent.function_trace()
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

    @newrelic.agent.function_trace()
    async def handle_quit(self, line):
        logger.info("QUIT command received")
        await self.write_response("221 Bye")

    @newrelic.agent.function_trace()
    async def handle_auth(self, line):
        logger.info(f"AUTH command received: {line}")
        # For now, accept any credentials
        await self.write_response("235 Authentication successful")

@newrelic.agent.background_task(name='smtp2x_heartbeat')
async def heartbeat(server):
    while True:
        if server.is_running:
            newrelic.agent.record_custom_event('smtp2xHeartbeat', {
                'status': 'running',
            })
            logger.info("Sent heartbeat to New Relic")
        else:
            newrelic.agent.record_custom_event('smtp2xHeartbeat', {
                'status': 'stopped',
            })
            logger.warning("smtp2x service is not running. Sent stopped status to New Relic")
            break  # Exit the heartbeat loop if the server is not running
        
        # Wait for 60 seconds before sending the next heartbeat
        await asyncio.sleep(60)

@newrelic.agent.background_task(name='smtp2x_main')
async def main():
    pagerduty_token = os.getenv("PAGERDUTY_API_TOKEN")
    pagerduty_service_id = os.getenv("PAGERDUTY_SERVICE_ID")
    pagerduty_from_email = os.getenv("PAGERDUTY_FROM_EMAIL")
    
    enabled_analyzers = os.getenv("ENABLED_ANALYZERS").lower().split(",")
    logger.info(f"Enabled analyzers: {enabled_analyzers}")
    
    config = SMTPConfig(
        hostname="192.168.40.191",
        max_message_size=50 * 1024 * 1024,  # 50 MB
        client_timeout=300,  # 5 minutes
        enabled_analyzers=enabled_analyzers,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        gpt4_task=os.getenv("GPT4_TASK"),
        gemini_task=os.getenv("GEMINI_TASK"),
        pagerduty_trigger=PagerDutyTrigger(pagerduty_token, pagerduty_service_id, pagerduty_from_email)
    )

    # Record custom event for server start
    newrelic.agent.record_custom_event('smtp2xStart', {
        'hostname': config.hostname,
        'max_message_size': config.max_message_size,
        'client_timeout': config.client_timeout
    })

    server = SMTPServer(config)
    
    # Create tasks for running the server and the heartbeat
    server_task = asyncio.create_task(server.start('0.0.0.0', 2525))
    heartbeat_task = asyncio.create_task(heartbeat(server))
    
    try:
        # Wait for both tasks concurrently
        await asyncio.gather(server_task, heartbeat_task)
    except Exception as e:
        logger.error(f"Error in smtp2x service or heartbeat: {str(e)}")
        newrelic.agent.record_exception()
        # Record custom event for server error
        newrelic.agent.record_custom_event('smtp2xError', {
            'error': str(e)
        })
    finally:
        server.stop()
        # Cancel the heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    # Set up New Relic application
    newrelic.agent.register_application(name='smtp2x')
    
    # Run the main function
    asyncio.run(main())
