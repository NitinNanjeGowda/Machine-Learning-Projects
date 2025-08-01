# Importing TextBlob to help the chatbot understand language nuances.
from textblob import TextBlob

# Defining the ChatBot class for interaction.
class ChatBot:
    def __init__(self):
        #Defining intents based on keywords
        self.intents = {
            "hours": {
                "keywords": ["hour", "open", "close"],
                "response": "We are open from 9 AM to 5 PM, Monday to Friday."
            },
            "return": {
                "keywords": ["refund", "money back", "return"],
                "response": "I'd be happy to help you process your return. Let me transfer you to a live agent."
            }
        }
    # Analyzing the sentiment of the user's message.
    def get_response(self, message):
        message = message.lower()
        for intent_data in self.intents.values():
            if any(word in message for word in intent_data["keywords"]):
                return intent_data["response"]
            
        # Generating the chatbot's response based on sentiment.
        sentiment = TextBlob(message).sentiment.polarity
        
        # Printing the chatbot's response and sentiment.
        return ("That's great to hear!" if sentiment > 0 else
                "I'm so sorry to hear that. How can I help?" if sentiment < 0 else
                "I see. Can you tell me more about that?")

    def chat(self):
        print("ChatBot: Hi, how can I help you today?")
        while (user_message := input("You: ").strip().lower()) not in ['exit', 'quit', 'bye']:
            print(f"\nChatBot: {self.get_response(user_message)}")
        print("ChatBot: Thank you for chatting. Have a great day!")

if __name__ == "__main__":
    # Creating the chatbot and starting the chat loop.
    ChatBot().chat()
