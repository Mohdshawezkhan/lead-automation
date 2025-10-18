"""
Lead Capture & Automation Workflow System - 100% FREE VERSION
==============================================================
Uses only free services with no charges:
- Groq (Free LLM API - faster than OpenAI)
- Google Sheets (Free)
- Google Calendar (Free)
- Discord Webhooks (Free alternative to Slack)
- Gmail SMTP (Free)
- Render.com / Railway (Free hosting)

Requirements:
pip install groq google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client gspread python-dotenv requests

Setup:
1. Get free Groq API key from https://console.groq.com
2. Set up Google credentials (always free)
3. Create Discord webhook (free Slack alternative)
4. Use Gmail for emails (free)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re

# Third-party imports
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from groq import Groq
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels - all FREE"""
    EMAIL = "email"
    DISCORD = "discord"  # Free alternative to Slack


@dataclass
class Lead:
    """Lead data structure"""
    name: str
    email: str
    phone: str
    car_model: str
    appointment_datetime: str
    intent_score: Optional[float] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ProcessedLead:
    """Processed lead with LLM analysis"""
    name: str
    phone: str
    model: str
    datetime: str
    intent_score: float


class LeadValidator:
    """Validates lead data"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        pattern = r'^\+?1?\d{9,15}$'
        return bool(re.match(pattern, phone.replace('-', '').replace(' ', '')))
    
    @staticmethod
    def validate_datetime(dt_string: str) -> bool:
        try:
            datetime.fromisoformat(dt_string)
            return True
        except ValueError:
            return False


class GroqProcessor:
    """
    Uses Groq API - 100% FREE with generous limits
    - 14,400 requests per day
    - Faster than OpenAI
    - No credit card required
    
    Get API key: https://console.groq.com
    """
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def analyze_lead(self, lead: Lead) -> ProcessedLead:
        """
        Send lead to Groq for analysis and get structured JSON response
        Model: llama-3.3-70b-versatile (FREE)
        """
        prompt = f"""
        Analyze this car dealership lead and return a strict JSON object with intent scoring.
        
        Lead Information:
        - Name: {lead.name}
        - Email: {lead.email}
        - Phone: {lead.phone}
        - Car Model: {lead.car_model}
        - Appointment: {lead.appointment_datetime}
        
        Calculate an intent_score (0.0 to 1.0) based on:
        - Email domain quality (corporate vs free email)
        - Car model (luxury vs economy)
        - Appointment timing (urgency)
        
        Return ONLY valid JSON in this exact format:
        {{
            "name": "string",
            "phone": "string",
            "model": "string",
            "datetime": "ISO8601 string",
            "intent_score": float
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # FREE model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a lead qualification assistant. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Groq analysis complete for {lead.name}")
            
            return ProcessedLead(**result)
            
        except Exception as e:
            logger.error(f"Groq processing failed: {e}")
            # Fallback to basic processing
            return ProcessedLead(
                name=lead.name,
                phone=lead.phone,
                model=lead.car_model,
                datetime=lead.appointment_datetime,
                intent_score=0.5
            )


class GoogleSheetsLogger:
    """
    Logs data to Google Sheets - 100% FREE
    No limits for personal use
    """
    
    def __init__(self, credentials_json: str, spreadsheet_name: str):
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        try:
            # Parse JSON string to dictionary
            import json
            creds_dict = json.loads(credentials_json)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse credentials JSON: {e}")
            raise
        self.client = gspread.authorize(creds)
        self.spreadsheet_name = spreadsheet_name
    
    def log_lead(self, lead: Lead, processed: ProcessedLead) -> bool:
        """Log lead data to Google Sheets"""
        try:
            sheet = self.client.open(self.spreadsheet_name).sheet1
            
            row = [
                lead.timestamp,
                processed.name,
                lead.email,
                processed.phone,
                processed.model,
                processed.datetime,
                processed.intent_score,
                "NEW"
            ]
            
            sheet.append_row(row)
            logger.info(f"Logged lead to Google Sheets: {processed.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log to Google Sheets: {e}")
            return False
    
    def get_all_leads(self) -> list:
        """Get all leads from sheet for dashboard"""
        try:
            sheet = self.client.open(self.spreadsheet_name).sheet1
            return sheet.get_all_records()
        except Exception as e:
            logger.error(f"Failed to get leads: {e}")
            return []


class CalendarManager:
    """
    Manages Google Calendar events - 100% FREE
    Includes Google Meet links (free)
    """
    
    def __init__(self, credentials_json: str):
            try:
                # Parse JSON string to dictionary
                creds_dict = json.loads(credentials_json)
                creds = Credentials.from_authorized_user_info(creds_dict)
                self.service = build('calendar', 'v3', credentials=creds)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse calendar credentials JSON: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize CalendarManager: {e}")
                raise
    
    def create_event(self, lead: Lead, processed: ProcessedLead) -> Optional[str]:
        """Create a Google Calendar event with FREE Meet link"""
        try:
            start_time = datetime.fromisoformat(processed.datetime)
            end_time = start_time + timedelta(hours=1)
            
            event = {
                'summary': f'Car Consultation - {processed.model}',
                'description': f'Lead consultation with {processed.name}\nIntent Score: {processed.intent_score}',
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'America/New_York',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'America/New_York',
                },
                'attendees': [
                    {'email': lead.email}
                ],
                'conferenceData': {
                    'createRequest': {
                        'requestId': f"lead-{datetime.now().timestamp()}",
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }
            
            event = self.service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1,
                sendUpdates='all'
            ).execute()
            
            meet_link = event.get('hangoutLink', '')
            logger.info(f"Calendar event created: {event.get('htmlLink')}")
            return meet_link
            
        except Exception as e:
            logger.error(f"Failed to create calendar event: {e}")
            return None


class NotificationService:
    """
    Sends notifications via FREE channels only
    - Discord (Free Slack alternative)
    - Gmail (Free email)
    """
    
    def __init__(self):
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        # Gmail is free - just use your account
        self.gmail_user = os.getenv('GMAIL_USER')
        self.gmail_password = os.getenv('GMAIL_APP_PASSWORD')  # App-specific password
    
    def send_notification(
        self,
        lead: Lead,
        processed: ProcessedLead,
        meet_link: Optional[str],
        channel: NotificationChannel
    ) -> bool:
        """Send notification through specified FREE channel"""
        
        if channel == NotificationChannel.DISCORD:
            return self._send_discord(lead, processed, meet_link)
        elif channel == NotificationChannel.EMAIL:
            return self._send_gmail(lead, processed, meet_link)
        
        return False
    
    def _send_discord(self, lead: Lead, processed: ProcessedLead, meet_link: Optional[str]) -> bool:
        """
        Send Discord notification - 100% FREE
        Better than Slack free tier (no message limit)
        
        Setup: Server Settings > Integrations > Webhooks > New Webhook
        """
        try:
            # Discord embed for rich formatting
            embed = {
                "title": f"🚗 New Lead: {processed.name}",
                "color": 5814783,  # Purple color
                "fields": [
                    {"name": "📧 Email", "value": lead.email, "inline": True},
                    {"name": "📱 Phone", "value": processed.phone, "inline": True},
                    {"name": "🚙 Model", "value": processed.model, "inline": True},
                    {"name": "📊 Intent Score", "value": f"{processed.intent_score:.2f}", "inline": True},
                    {"name": "📅 Appointment", "value": processed.datetime, "inline": False}
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Lead Automation System"}
            }
            
            if meet_link:
                embed["fields"].append({
                    "name": "🎥 Google Meet",
                    "value": f"[Join Meeting]({meet_link})",
                    "inline": False
                })
            
            message = {
                "content": f"@here New lead received!",
                "embeds": [embed]
            }
            
            response = requests.post(self.discord_webhook, json=message)
            response.raise_for_status()
            logger.info("Discord notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    def _send_gmail(self, lead: Lead, processed: ProcessedLead, meet_link: Optional[str]) -> bool:
        """
        Send email via Gmail - 100% FREE
        Limit: 500 emails/day (more than enough)
        
        Setup: Enable 2FA, create App Password
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.gmail_user
            msg['To'] = lead.email
            msg['Subject'] = f"✅ Appointment Confirmed - {processed.model}"
            
            # HTML email template
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                    .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                    .button {{ display: inline-block; padding: 12px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                    .info-box {{ background: white; padding: 20px; border-left: 4px solid #667eea; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚗 Appointment Confirmed!</h1>
                    </div>
                    <div class="content">
                        <p>Dear {processed.name},</p>
                        <p>Your consultation appointment has been successfully scheduled!</p>
                        
                        <div class="info-box">
                            <h3>Appointment Details:</h3>
                            <p><strong>🚙 Car Model:</strong> {processed.model}</p>
                            <p><strong>📅 Date & Time:</strong> {processed.datetime}</p>
                            <p><strong>📱 Phone:</strong> {processed.phone}</p>
                        </div>
                        
                        {f'<a href="{meet_link}" class="button">Join Google Meet</a>' if meet_link else ''}
                        
                        <p>We look forward to seeing you!</p>
                        <p>If you need to reschedule, please contact us at least 24 hours in advance.</p>
                        
                        <p>Best regards,<br><strong>Your Dealership Team</strong></p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Plain text fallback
            text = f"""
            Dear {processed.name},
            
            Your appointment has been confirmed!
            
            Details:
            - Car Model: {processed.model}
            - Date & Time: {processed.datetime}
            - Phone: {processed.phone}
            
            {f'Join Meeting: {meet_link}' if meet_link else ''}
            
            We look forward to seeing you!
            
            Best regards,
            Your Dealership Team
            """
            
            msg.attach(MIMEText(text, 'plain'))
            msg.attach(MIMEText(html, 'html'))
            
            # Send via Gmail SMTP (FREE)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Gmail sent to {lead.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Gmail: {e}")
            return False


class LeadWorkflow:
    """Main workflow orchestrator - 100% FREE services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.validator = LeadValidator()
        self.groq_processor = GroqProcessor(config['groq_api_key'])
        self.sheets_logger = GoogleSheetsLogger(
            config['google_sheets_credentials'],
            config['spreadsheet_name']
        )
        self.calendar_manager = CalendarManager(config['google_calendar_credentials'])
        self.notification_service = NotificationService()
        self.notification_channels = config.get(
            'notification_channels',
            [NotificationChannel.DISCORD, NotificationChannel.EMAIL]
        )
    
    def process_lead(self, lead_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a lead through the complete workflow
        
        Args:
            lead_data: Dictionary with keys: name, email, phone, car_model, appointment_datetime
            
        Returns:
            Dictionary with processing results and status
        """
        result = {
            'success': False,
            'lead_id': None,
            'errors': [],
            'meet_link': None
        }
        
        try:
            # Step 1: Validate input
            if not self.validator.validate_email(lead_data['email']):
                result['errors'].append("Invalid email format")
                return result
            
            if not self.validator.validate_phone(lead_data['phone']):
                result['errors'].append("Invalid phone format")
                return result
            
            if not self.validator.validate_datetime(lead_data['appointment_datetime']):
                result['errors'].append("Invalid datetime format")
                return result
            
            # Step 2: Create Lead object
            lead = Lead(**lead_data)
            logger.info(f"Processing lead: {lead.name}")
            
            # Step 3: Groq Analysis (FREE)
            processed_lead = self.groq_processor.analyze_lead(lead)
            logger.info(f"Intent score: {processed_lead.intent_score}")
            
            # Step 4: Log to Google Sheets (FREE)
            sheets_success = self.sheets_logger.log_lead(lead, processed_lead)
            if not sheets_success:
                result['errors'].append("Failed to log to Google Sheets")
            
            # Step 5: Create Calendar Event (FREE with Meet link)
            meet_link = self.calendar_manager.create_event(lead, processed_lead)
            result['meet_link'] = meet_link
            if not meet_link:
                result['errors'].append("Failed to create calendar event")
            
            # Step 6: Send Notifications (All FREE)
            for channel in self.notification_channels:
                notification_success = self.notification_service.send_notification(
                    lead, processed_lead, meet_link, channel
                )
                if not notification_success:
                    result['errors'].append(f"Failed to send {channel.value} notification")
            
            result['success'] = True
            result['lead_id'] = lead.timestamp
            logger.info(f"Lead processing complete: {lead.name}")
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            result['errors'].append(str(e))
        
        return result


# Flask API for webhook integration (Deploy FREE on Render/Railway)
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration - all FREE services
config = {
    'groq_api_key': os.getenv('GROQ_API_KEY'),  # FREE from console.groq.com
    'google_sheets_credentials': os.getenv('GOOGLE_SHEETS_CREDS', './credentials/sheets_creds.json'),
    'google_calendar_credentials': os.getenv('GOOGLE_CALENDAR_CREDS', './credentials/calendar_creds.json'),
    'spreadsheet_name': os.getenv('SPREADSHEET_NAME', 'Lead Tracker'),
    'notification_channels': [NotificationChannel.DISCORD, NotificationChannel.EMAIL]
}

# Initialize workflow
workflow = LeadWorkflow(config)

@app.route('/webhook/lead', methods=['POST'])
def capture_lead():
    """
    Webhook endpoint for lead capture
    FREE to host on: Render.com, Railway.app, or Fly.io
    """
    try:
        lead_data = request.json
        result = workflow.process_lead(lead_data)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/dashboard', methods=['GET'])
def dashboard_data():
    """Get all leads for dashboard"""
    try:
        leads = workflow.sheets_logger.get_all_leads()
        return jsonify(leads), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # For local development
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)