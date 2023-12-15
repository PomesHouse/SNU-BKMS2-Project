#서버 두개켜서 확인한다
import os
# Use the package we installed
from slack_bolt import App

# Initialize your app with your bot token and signing secret
app = App(
    token="xoxb-6157885647652-6363066331953-PXAsCaUR4505uM0O6vEBqtE2",
    signing_secret="5dac6e18c85d42931f5747a81377120f"
)

# Add functionality here later
# @app.event("app_home_opened") etc.

# Ready? Start your app!
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 5000)))
    