"""
Alert Management Module
Handles detection processing, alert level determination, and notifications
PLACEHOLDER for future SMS/Call integration
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class DetectionEvent:
    """Represents a single detection event"""
    timestamp: str
    confidence: float
    alert_level: int
    duration: float
    class_name: str
    bbox: Optional[List[float]] = None


class AlertManager:
    """Manages alert levels and notifications"""
    
    def __init__(self, alert_config: Dict[str, Any], notification_config: Dict[str, Any]):
        """
        Initialize alert manager
        
        Args:
            alert_config: Alert configuration from config.yaml
            notification_config: Notification configuration from config.yaml
        """
        self.alert_config = alert_config
        self.notification_config = notification_config
        
        # Alert thresholds
        self.level_1_min = alert_config['LEVEL_1']['MIN_CONFIDENCE']
        self.level_1_max = alert_config['LEVEL_1']['MAX_CONFIDENCE']
        self.level_2_min = alert_config['LEVEL_2']['MIN_CONFIDENCE']
        self.level_2_critical = alert_config['LEVEL_2']['CRITICAL_CONFIDENCE']
        self.level_2_duration = alert_config['LEVEL_2']['DURATION_THRESHOLD']
        
        # Duration tracking
        self.drowning_start_time = None
        self.continuous_drowning_frames = 0
        
        # Detection history
        self.latest_detections = []
        self.detection_history = []
    
    def process_detection(self, results, current_time: float) -> Optional[DetectionEvent]:
        """
        Process detection results and determine alert level
        
        Args:
            results: YOLO detection results
            current_time: Current timestamp
        
        Returns:
            DetectionEvent if alert triggered, None otherwise
        """
        detections = results.boxes
        
        if len(detections) == 0:
            # Reset duration tracking when no detections
            self.drowning_start_time = None
            self.continuous_drowning_frames = 0
            return None
        
        # Find drowning detections (class 0)
        drowning_count = sum(1 for box in detections if int(box.cls[0]) == 0)
        
        if drowning_count == 0:
            self.drowning_start_time = None
            self.continuous_drowning_frames = 0
            return None
        
        # Get highest confidence drowning detection
        drowning_boxes = [box for box in detections if int(box.cls[0]) == 0]
        max_conf = max(float(box.conf[0]) for box in drowning_boxes)
        conf_percentage = round(max_conf * 100, 2)
        
        # Track continuous drowning duration
        if self.drowning_start_time is None:
            self.drowning_start_time = current_time
            self.continuous_drowning_frames = 1
            duration = 0.0
        else:
            duration = current_time - self.drowning_start_time
            self.continuous_drowning_frames += 1
        
        # Determine alert level
        alert_level = self._determine_alert_level(max_conf, duration)
        
        if alert_level > 0:
            event = DetectionEvent(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                confidence=conf_percentage,
                alert_level=alert_level,
                duration=duration,
                class_name="Drowning"
            )
            
            # Add to history
            self.detection_history.append(event)
            
            # Trigger notifications (if enabled)
            self._send_notification(event)
            
            return event
        
        return None
    
    def _determine_alert_level(self, confidence: float, duration: float) -> int:
        """
        Determine alert level based on confidence and duration
        
        Returns:
            0: No alert
            1: Level 1 Warning
            2: Level 2 Emergency
        """
        # Level 2 (Emergency) conditions
        if confidence >= self.level_2_critical:
            # Instant Level 2: 80%+ confidence
            return 2
        elif confidence >= self.level_2_min and duration >= self.level_2_duration:
            # Level 2: 65%+ confidence for 3+ seconds
            return 2
        
        # Level 1 (Warning) conditions
        elif self.level_1_min <= confidence <= self.level_1_max:
            # Level 1: 50-64% confidence
            return 1
        
        return 0
    
    def _send_notification(self, event: DetectionEvent):
        """
        Send notification for detection event
        PLACEHOLDER for future SMS/Call integration
        
        Args:
            event: Detection event to send notification for
        """
        # Check if notifications are enabled
        if not self.notification_config.get('SMS_ENABLED', False) and \
           not self.notification_config.get('CALL_ENABLED', False):
            return
        
        # FUTURE IMPLEMENTATION:
        # - Send SMS for Level 1 and Level 2
        # - Initiate phone call for Level 2
        # - Use configured phone numbers from config
        
        print(f"📱 [PLACEHOLDER] Notification would be sent:")
        print(f"   Level: {event.alert_level}")
        print(f"   Confidence: {event.confidence}%")
        print(f"   Duration: {event.duration}s")
        
        # Future integration points:
        # if event.alert_level == 1:
        #     self._send_sms(event)
        # elif event.alert_level == 2:
        #     self._send_sms(event)
        #     self._initiate_call(event)
    
    def _send_sms(self, event: DetectionEvent):
        """
        Send SMS notification
        PLACEHOLDER for future Twilio/AWS SNS/Vonage integration
        """
        # Get SMS configuration
        sms_config = self.notification_config.get('SMS_PROVIDER', {})
        recipient = self.notification_config.get('SMS_RECIPIENT', {})
        
        # Get message template
        messages = self.notification_config.get('MESSAGES', {})
        if event.alert_level == 1:
            message = messages.get('SMS_LEVEL_1', '')
        else:
            message = messages.get('SMS_LEVEL_2', '')
        
        # Format message
        message = message.format(
            confidence=event.confidence,
            timestamp=event.timestamp,
            duration=event.duration
        )
        
        # TODO: Implement actual SMS sending
        # Example with Twilio:
        # from twilio.rest import Client
        # client = Client(sms_config['API_KEY'], sms_config['API_SECRET'])
        # message = client.messages.create(
        #     body=message,
        #     from_=sms_config['SENDER_ID'],
        #     to=recipient['NUMBER']
        # )
        
        print(f"📤 [SMS PLACEHOLDER] To: {recipient.get('NUMBER', 'NOT_CONFIGURED')}")
        print(f"   Message: {message}")
    
    def _initiate_call(self, event: DetectionEvent):
        """
        Initiate emergency phone call
        PLACEHOLDER for future Twilio/Vonage integration
        """
        # Get call configuration
        call_config = self.notification_config.get('CALL_PROVIDER', {})
        emergency = self.notification_config.get('EMERGENCY_CALL', {})
        
        # Get call message
        messages = self.notification_config.get('MESSAGES', {})
        message = messages.get('CALL_MESSAGE', '')
        
        # TODO: Implement actual call initiation
        # Example with Twilio:
        # from twilio.rest import Client
        # client = Client(call_config['API_KEY'], call_config['API_SECRET'])
        # call = client.calls.create(
        #     twiml=f'<Response><Say>{message}</Say></Response>',
        #     to=emergency['NUMBER'],
        #     from_=call_config['CALLER_ID']
        # )
        
        print(f"📞 [CALL PLACEHOLDER] To: {emergency.get('NUMBER', 'NOT_CONFIGURED')}")
        print(f"   Authority: {emergency.get('NAME', 'Emergency Contact')}")
        print(f"   Message: {message}")
    
    def get_detection_history(self) -> List[DetectionEvent]:
        """Get all detection events"""
        return self.detection_history
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history.clear()
        self.latest_detections.clear()
    
    def reset_duration_tracking(self):
        """Reset duration tracking"""
        self.drowning_start_time = None
        self.continuous_drowning_frames = 0
