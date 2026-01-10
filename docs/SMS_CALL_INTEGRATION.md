# 📞 SMS & Call Integration Guide

## Overview
This guide explains how to integrate SMS and phone call notifications when drowning alerts are detected.

**Status**: 🚧 Not yet implemented - Placeholders ready in `src/alerts/alert_manager.py`

---

## Supported Providers

### 1. Twilio (Recommended)
- ✅ SMS and voice calls
- ✅ Reliable and well-documented
- ✅ Pay-as-you-go pricing
- ✅ Trial account available

### 2. AWS SNS (Amazon Simple Notification Service)
- ✅ SMS only
- ✅ AWS ecosystem integration
- ✅ Scalable
- ⚠️ Requires AWS account

### 3. Vonage (formerly Nexmo)
- ✅ SMS and voice calls
- ✅ Global coverage
- ✅ Competitive pricing

---

## Configuration

### Step 1: Choose Provider
Edit `config/config.yaml`:

```yaml
NOTIFICATIONS:
  SMS_ENABLED: true
  CALL_ENABLED: true
  
  # Phone Numbers (E.164 format)
  SMS_RECIPIENT:
    NUMBER: "+1234567890"      # Replace with actual number
    NAME: "Pool Manager"
  
  EMERGENCY_CALL:
    NUMBER: "+1987654321"      # Replace with emergency contact
    NAME: "Security Office"
  
  # Provider selection
  SMS_PROVIDER:
    SERVICE: "twilio"           # or "aws_sns" or "vonage"
    API_KEY: ""                 # From .env
    API_SECRET: ""              # From .env
    SENDER_ID: ""               # Your Twilio phone number
```

### Step 2: Add Credentials
Copy `.env.template` to `.env`:

```bash
cp config/.env.template config/.env
```

Edit `.env` with your credentials:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890

# Or AWS SNS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Or Vonage
VONAGE_API_KEY=your_api_key
VONAGE_API_SECRET=your_api_secret
VONAGE_PHONE_NUMBER=+1234567890
```

**⚠️ NEVER commit `.env` to version control!**

---

## Twilio Integration (Recommended)

### Step 1: Create Twilio Account
1. Go to https://www.twilio.com/try-twilio
2. Sign up for free trial (includes $15 credit)
3. Get a phone number
4. Find your Account SID and Auth Token in console

### Step 2: Install Twilio
```bash
pip install twilio
```

### Step 3: Implement SMS Function
Edit `src/alerts/alert_manager.py`, uncomment SMS implementation:

```python
def _send_sms(self, event: DetectionEvent):
    """Send SMS notification via Twilio"""
    from twilio.rest import Client
    
    # Get configuration
    sms_config = self.notification_config['SMS_PROVIDER']
    recipient = self.notification_config['SMS_RECIPIENT']
    
    # Get message template
    messages = self.notification_config['MESSAGES']
    if event.alert_level == 1:
        message = messages['SMS_LEVEL_1']
    else:
        message = messages['SMS_LEVEL_2']
    
    # Format message
    message = message.format(
        confidence=event.confidence,
        timestamp=event.timestamp,
        duration=event.duration
    )
    
    # Send SMS
    try:
        client = Client(sms_config['API_KEY'], sms_config['API_SECRET'])
        sent_message = client.messages.create(
            body=message,
            from_=sms_config['SENDER_ID'],
            to=recipient['NUMBER']
        )
        print(f"✅ SMS sent: {sent_message.sid}")
    except Exception as e:
        print(f"❌ SMS failed: {e}")
```

### Step 4: Implement Call Function
```python
def _initiate_call(self, event: DetectionEvent):
    """Initiate emergency phone call via Twilio"""
    from twilio.rest import Client
    
    # Get configuration
    call_config = self.notification_config['CALL_PROVIDER']
    emergency = self.notification_config['EMERGENCY_CALL']
    
    # Get call message
    messages = self.notification_config['MESSAGES']
    call_message = messages['CALL_MESSAGE']
    
    # Create TwiML for voice message
    twiml = f'''
    <Response>
        <Say voice="Polly.Amy">
            Emergency drowning alert detected at monitored pool location.
            Confidence level: {event.confidence} percent.
            Duration: {event.duration} seconds.
            Immediate response required.
        </Say>
        <Pause length="1"/>
        <Say>This message will repeat.</Say>
        <Pause length="2"/>
        <Redirect/>
    </Response>
    '''
    
    # Initiate call
    try:
        client = Client(call_config['API_KEY'], call_config['API_SECRET'])
        call = client.calls.create(
            twiml=twiml,
            to=emergency['NUMBER'],
            from_=call_config['CALLER_ID']
        )
        print(f"✅ Call initiated: {call.sid}")
    except Exception as e:
        print(f"❌ Call failed: {e}")
```

### Step 5: Test
```python
# Test SMS
python -c "from src.alerts import AlertManager, DetectionEvent; \
from src.core import get_config; \
config = get_config(); \
mgr = AlertManager(config.get_alert_config(), config.get_notification_config()); \
event = DetectionEvent('2026-01-10 12:00:00', 85.0, 2, 5.0, 'Drowning'); \
mgr._send_sms(event)"
```

---

## AWS SNS Integration

### Step 1: Setup AWS Account
1. Create AWS account
2. Enable SNS service
3. Create SNS topic for alerts
4. Get access key and secret

### Step 2: Install boto3
```bash
pip install boto3
```

### Step 3: Implement SMS Function
```python
def _send_sms(self, event: DetectionEvent):
    """Send SMS via AWS SNS"""
    import boto3
    
    sms_config = self.notification_config['SMS_PROVIDER']
    recipient = self.notification_config['SMS_RECIPIENT']
    
    # Format message
    messages = self.notification_config['MESSAGES']
    if event.alert_level == 1:
        message = messages['SMS_LEVEL_1']
    else:
        message = messages['SMS_LEVEL_2']
    
    message = message.format(
        confidence=event.confidence,
        timestamp=event.timestamp,
        duration=event.duration
    )
    
    # Send via SNS
    try:
        sns_client = boto3.client(
            'sns',
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=sms_config['API_KEY'],
            aws_secret_access_key=sms_config['API_SECRET']
        )
        
        response = sns_client.publish(
            PhoneNumber=recipient['NUMBER'],
            Message=message,
            MessageAttributes={
                'AWS.SNS.SMS.SMSType': {
                    'DataType': 'String',
                    'StringValue': 'Transactional'  # For critical alerts
                }
            }
        )
        print(f"✅ SMS sent via AWS SNS: {response['MessageId']}")
    except Exception as e:
        print(f"❌ AWS SNS failed: {e}")
```

---

## Notification Flow

### Alert Level 1 (Warning)
```
Detection (50-64% confidence)
        ↓
AlertManager.process_detection()
        ↓
Alert Level 1 determined
        ↓
_send_sms() ← SMS only for Level 1
        ↓
SMS sent to SMS_RECIPIENT_NUMBER
```

### Alert Level 2 (Emergency)
```
Detection (65%+ or 3+ seconds)
        ↓
AlertManager.process_detection()
        ↓
Alert Level 2 determined
        ↓
├─→ _send_sms() ← SMS to SMS_RECIPIENT_NUMBER
└─→ _initiate_call() ← Call to EMERGENCY_CALL_NUMBER
        ↓
SMS + Voice call sent simultaneously
```

---

## Message Templates

### Customize Messages
Edit `config/config.yaml`:

```yaml
NOTIFICATIONS:
  MESSAGES:
    SMS_LEVEL_1: |
      ⚠️ POOL ALERT
      Unsafe movement detected
      Confidence: {confidence}%
      Time: {timestamp}
      Please investigate immediately.
    
    SMS_LEVEL_2: |
      🚨 DROWNING EMERGENCY
      Location: Main Pool
      Confidence: {confidence}%
      Duration: {duration}s
      IMMEDIATE ACTION REQUIRED!
      
    CALL_MESSAGE: |
      Emergency drowning alert detected at main pool area.
      Immediate lifeguard response required.
      Check pool cameras for verification.
```

### Available Placeholders
- `{confidence}` - Detection confidence percentage
- `{timestamp}` - Detection timestamp
- `{duration}` - Duration in seconds
- `{alert_level}` - 1 or 2

---

## Testing

### Test SMS Without Detection
```python
from src.alerts import AlertManager, DetectionEvent
from src.core import get_config

config = get_config()
alert_mgr = AlertManager(
    config.get_alert_config(),
    config.get_notification_config()
)

# Create test event
test_event = DetectionEvent(
    timestamp="2026-01-10 14:30:00",
    confidence=75.5,
    alert_level=2,
    duration=4.2,
    class_name="Drowning"
)

# Send test SMS
alert_mgr._send_sms(test_event)

# Initiate test call
alert_mgr._initiate_call(test_event)
```

### Test with Dashboard
1. Start dashboard: `python app.py`
2. Enable webcam detection
3. Trigger alert (manually or with test video)
4. Verify SMS/call received

---

## Troubleshooting

### SMS Not Sending

**Check 1**: Verify credentials
```python
from twilio.rest import Client
client = Client('account_sid', 'auth_token')
print(client.api.accounts.list())  # Should not error
```

**Check 2**: Verify phone number format
- Must be E.164 format: `+[country code][number]`
- Example: `+12025551234` (USA)

**Check 3**: Check Twilio trial restrictions
- Trial accounts can only send to verified numbers
- Upgrade to paid account for production use

### Call Not Working

**Issue**: Call drops immediately
- Check TwiML syntax
- Verify voice parameter (`Polly.Amy`, `alice`, etc.)
- Test TwiML in Twilio console first

**Issue**: Wrong number called
- Verify `EMERGENCY_CALL_NUMBER` in config
- Check `.env` override values

### General Issues

**Config not loading**:
```python
from src.core import get_config
config = get_config()
print(config.get_notification_config())  # Verify values
```

**Import errors**:
```bash
pip install twilio  # or boto3, vonage
python -c "import twilio"  # Test import
```

---

## Cost Estimates

### Twilio (Pay-as-you-go)
- SMS (US): $0.0079 per message
- Voice (US): $0.013 per minute
- Example: 100 alerts/month = ~$2/month

### AWS SNS
- SMS (US): $0.00645 per message
- No voice support
- Example: 100 alerts/month = ~$0.65/month

### Vonage
- SMS (US): $0.0076 per message
- Voice (US): $0.012 per minute
- Similar to Twilio pricing

---

## Security Best Practices

1. **Never hard-code credentials**
   - Use `.env` file
   - Add `.env` to `.gitignore`

2. **Rotate API keys regularly**
   - Update in Twilio/AWS console
   - Update `.env` file

3. **Limit API permissions**
   - Twilio: Only SMS/Voice permissions
   - AWS: Only SNS publish permissions

4. **Monitor usage**
   - Set up usage alerts in provider console
   - Prevent accidental cost overruns

5. **Test in development**
   - Use sandbox/test numbers
   - Don't spam real emergency numbers

---

## Production Checklist

- [ ] Provider account created and verified
- [ ] API credentials obtained and tested
- [ ] Phone numbers verified and in E.164 format
- [ ] `.env` file created (not committed)
- [ ] config.yaml updated with correct numbers
- [ ] SMS function implemented and tested
- [ ] Call function implemented and tested
- [ ] Message templates customized
- [ ] Error handling added
- [ ] Logging configured
- [ ] Usage monitoring enabled
- [ ] Tested with real detections

---

## Example Full Integration

```python
# src/alerts/alert_manager.py (full implementation)

def _send_notification(self, event: DetectionEvent):
    """Send notifications based on alert level"""
    
    # Check if notifications enabled
    if not self.notification_config.get('SMS_ENABLED') and \
       not self.notification_config.get('CALL_ENABLED'):
        return
    
    print(f"📱 Sending notifications for Alert Level {event.alert_level}")
    
    # Level 1: SMS only
    if event.alert_level == 1:
        if self.notification_config.get('SMS_ENABLED'):
            self._send_sms(event)
    
    # Level 2: SMS + Call
    elif event.alert_level == 2:
        if self.notification_config.get('SMS_ENABLED'):
            self._send_sms(event)
        if self.notification_config.get('CALL_ENABLED'):
            self._initiate_call(event)
```

---

**Ready to integrate?** Start with Twilio SMS, then add voice calls for Level 2 emergencies!
