#https://www.pragnakalp.com/create-slack-bot-using-python-tutorial-with-examples/
#test.py -> 터미널 하나더 ngrok http 5000 -> slack event subscription url 바꾸기 save까지
#https://api.slack.com/apps/A06A3P8H4EA/event-subscriptions?
import slack
from flask import Flask
from slackeventsapi import SlackEventAdapter
 
SLACK_TOKEN="xoxb-6157885647652-6363066331953-PXAsCaUR4505uM0O6vEBqtE2"
SIGNING_SECRET="5dac6e18c85d42931f5747a81377120f"
 
app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)
 
client = slack.WebClient(token=SLACK_TOKEN)
 
@ slack_event_adapter.on('message')
def message(payload):
    print(payload)
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
 
    if text == "hi":
        print("got it!")
        client.chat_postMessage(channel=channel_id,text="Hello")
 
if __name__ == "__main__":
    app.run(debug=True)