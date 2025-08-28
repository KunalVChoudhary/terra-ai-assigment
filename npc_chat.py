import datetime
from typing import List
from dataclasses import dataclass
from collections import defaultdict
import json
import ollama   # ✅ use ollama client


@dataclass
class PlayerMessage:  #players msg info
    player_id: int
    text: str
    timestamp: datetime.datetime
    

@dataclass
class NPCResponse:  #ai response
    player_id: int
    message_text: str
    npc_reply: str
    conversation_state: List[str]
    npc_mood: str
    timestamp: datetime.datetime

class OpenAINPCChatSystem:
    def __init__(self, model="llama3"):   # ✅ no api_key needed
        self.model = model
        self.player_conversation_history = defaultdict(list)
        self.player_moods = defaultdict(lambda: "neutral")
        self.responses = []

        self.friendly_keywords = [
            'hello', 'hi', 'thank', 'please', 'help', 'appreciate', 'kind',
            'wonderful', 'great', 'best', 'grateful', 'thanks', 'good'
        ]

        self.angry_keywords = [
            'useless', 'hate', 'terrible', 'worst', 'stupid', 'frustrated',
            'angry', 'broken', 'incompetent', 'awful', 'horrible', 'sucks'
        ]

    def detect_mood(self, message_text, current_mood):
        text_lower = message_text.lower()
        friendly_score = sum(1 for keyword in self.friendly_keywords if keyword in text_lower)
        angry_score = sum(1 for keyword in self.angry_keywords if keyword in text_lower)

        if angry_score > friendly_score and angry_score > 0:
            return "angry"
        elif friendly_score > angry_score and friendly_score > 0:
            return "friendly"
        else:
            if current_mood != "neutral":
                return "neutral"
            return current_mood

    def generate_reply_ollama(self, player_id, message, conversation_state, mood):
        previous_msg = ""
        if conversation_state:
            for i, prev_msg in enumerate(conversation_state[-3:], 1):
                previous_msg += f"{i}. Player: \"{prev_msg}\"\n"

        prompt = f"""
        You are an NPC (Non-Playable Character) in a role-playing game. 
        Your job is to respond in a way that feels immersive and natural.

        Context:
        - Player's current mood: {mood} (Player {player_id})
        - Previous interactions:
        {previous_msg}
        - Player {player_id} just said: "{message}"

        Respond in character as the NPC. Keep your response small (1-2 sentences maximum).
        """

        try:
            client = ollama.Client(host="http://localhost:11434")  #ollama serve
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful NPC in a fantasy game. Keep responses brief and in character."},
                    {"role": "user", "content": prompt}
                ],
            )
            content = response['message']['content'] if 'message' in response else response
            return content

        except Exception as e:
            print(e)
            return self.default_response()

    def default_response(self):
        return 'Issue with ollama response'

    def process_messages(self, player_json):
        with open(player_json, 'r') as f:
            json_messages = json.load(f)

        messages = [
            PlayerMessage(
                player_id=msg["player_id"],
                text=msg["text"],
                timestamp=datetime.datetime.fromisoformat(msg["timestamp"])
            )
            for msg in json_messages
        ]
        messages.sort(key=lambda x: x.timestamp)

        for i, message in enumerate(messages, 1):
            conversation_state = self.player_conversation_history[message.player_id][-3:]
            current_mood = self.player_moods[message.player_id]
            new_mood = self.detect_mood(message.text, current_mood)
            self.player_moods[message.player_id] = new_mood

            gpt_reply = self.generate_reply_ollama(
                message.player_id, 
                message.text, 
                conversation_state, 
                new_mood
            )
            self.player_conversation_history[message.player_id].append(message.text)

            response = NPCResponse(
                player_id=message.player_id,
                message_text=message.text,
                npc_reply=gpt_reply,
                conversation_state=conversation_state.copy(),
                npc_mood=new_mood,
                timestamp=message.timestamp
            )

            self.responses.append(response)
            self.print_reply(i, response)
    
    def print_reply(self, i, response):
        print(f"Message {i}")
        print(f"Player ID: {response.player_id}")
        print(f"Timestamp: {response.timestamp.isoformat()}")
        print(f"Player Message: '{response.message_text}'")
        print(f"NPC Reply: '{response.npc_reply}'")
        print(f"NPC Mood: {response.npc_mood}")
        print(f"Conversation State: {response.conversation_state}")
        print(' ')
        print(' ')    

    def save_chat_json(self):
        chat_history = []
        for response in self.responses:
            chat_history.append(
                {"player_id": response.player_id,
                 "timestamp": response.timestamp.isoformat(),
                 "player_message": response.message_text,
                 "npc_reply": response.npc_reply,
                 "npc_mood": response.npc_mood,
                 "conversation_state": response.conversation_state}
            )
        with open('ollama_npc_chat_history.json', 'w') as f:
            json.dump(chat_history, f, indent=2)
        print('Saved chat history to ollama_npc_chat_history.json')

def main():

    npc_gpt_chat=OpenAINPCChatSystem()
    npc_gpt_chat.process_messages('players.json')
    npc_gpt_chat.save_chat_json()

if __name__ == "__main__":
    main()
