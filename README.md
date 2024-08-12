# SMTP Server with Image Analysis

This project implements a custom SMTP server with integrated image analysis capabilities using GPT-4 Vision. It's designed to process incoming emails, analyze attached images, and trigger alerts based on the analysis results.

## Features

- Custom SMTP server implementation
- Image analysis using OpenAI's GPT-4 Vision API
- Telegram integration for sending critical images
- PagerDuty integration for incident management
- Docker support for easy deployment

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Telegram Bot Token and Chat ID
- PagerDuty API Token and Service ID

## Configuration

Set up the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `TELEGRAM_BOT_TOKEN`: Your Telegram Bot Token
- `TELEGRAM_CHAT_ID`: Your Telegram Chat ID
- `GPT4_TASK`: The task description for GPT-4 Vision
- `PAGERDUTY_FROM_EMAIL`: Email address for PagerDuty notifications
- `PAGERDUTY_API_TOKEN`: Your PagerDuty API Token
- `PAGERDUTY_SERVICE_ID`: Your PagerDuty Service ID

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/YogevKr/smtp2x.git
   cd smtp2x
   ```

2. Create a `.env` file in the project root and add your environment variables.

3. Build and run the Docker container:
   ```
   docker-compose up --build
   ```

4. The SMTP server will be available on port 2525.

## How it Works

1. The SMTP server receives emails with image attachments.
2. Attached images are analyzed using GPT-4 Vision.
3. If the analysis detects a critical image, it's sent to a Telegram chat.
4. A PagerDuty incident is created for critical images.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
