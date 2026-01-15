class AlertSystem:
    def __init__(self, method="console"):
        self.method = method

    def send_alert(self, message):
        if self.method == "console":
            print(f"[ALERT] {message}")
        elif self.method == "email":
            print(f"[EMAIL ALERT] {message}")  # Placeholder for email sending
        else:
            print(f"[UNKNOWN ALERT] {message}")
