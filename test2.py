import requests
import json


class OllamaChat():
    def __init__(self, system_message="你的名字叫做minglog，是一个由骆明开发的大语言模型。",
                 url="http://localhost:6006/api/chat", model_name="deepseek-r1:7b"):
        """
            url: ChatModel API. Default is Local Ollama
                # url = "http://localhost:6006/api/chat"  # AutoDL
                # url = "http://localhost:11434/api/chat"  # localhost
            model_name: ModelName.
                Default is Qwen:7b
        """
        self.url = url
        self.model_name = model_name
        self.system_message = {
            "role": "system",
            "content": f"""{system_message}"""
        }
        self.message = [self.system_message]

    def __ouput_response(self, response, stream=False, is_chat=True):
        if stream:
            return_text = ''
            # 流式接收输出
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    print(decoded_chunk)  # 打印响应内容以检查实际结构
                    try:
                        data = json.loads(decoded_chunk)
                        if is_chat:
                            text = data['message']['content'] if 'message' in data else ''
                        else:
                            text = data['response'] if 'response' in data else ''
                    except json.JSONDecodeError:
                        text = '无法解析返回数据'
                    except KeyError:
                        text = '返回数据缺少预期的键'
                    return_text += text
                    print(text, end='', flush=True)
        else:
            try:
                data = json.loads(response.text)
                if is_chat:
                    return_text = ''.join(
                        [data['message']['content'] for response in response.text.split('\n') if len(response) != 0])
                else:
                    return_text = ''.join(
                        [data['response'] for response in response.text.split('\n') if len(response) != 0])
            except json.JSONDecodeError:
                return_text = '无法解析返回数据'
            except KeyError:
                return_text = '返回数据缺少预期的键'

        return return_text

    def chat(self, prompt, message=None, stream=False, system_message=None, **options):
        """
            prompt: Type Str, User input prompt words.
            messages: Type List, Dialogue History. role in [system, user, assistant]
            stream: Type Boolean, Is it streaming output. if `True` streaming output, otherwise not streaming output.
            system_message: Type Str, System Prompt. Default self.system_message.
            **options: option items.
        """
        if message is not None:
            self.message = message
        if message == []:
            self.message.append(self.system_message)
        if system_message:
            self.message[0]['content'] = system_message
        self.message.append({"role": "user", "content": prompt})
        if 'max_tokens' in options:
            options['num_ctx'] = options['max_tokens']
        data = {
            "model": self.model_name,
            "messages": self.message,
            "options": options
        }
        headers = {
            "Content-Type": "application/json"
        }
        responses = requests.post(self.url, headers=headers, json=data, stream=stream)
        return_text = self.__ouput_response(responses, stream)
        self.message.append({"role": "assistant", "content": return_text})
        return return_text, self.message

    def generate(self, prompt, stream=False, **options):
        generate_url = self.url.replace('chat', 'generate')
        if 'max_tokens' in options:
            options['num_ctx'] = options['max_tokens']
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "options": options
        }
        headers = {
            "Content-Type": "application/json"
        }
        responses = requests.post(generate_url, headers=headers, json=data, stream=stream)
        return_text = self.__ouput_response(responses, stream, is_chat=False)
        return return_text


if __name__ == "__main__":
    Chat = OllamaChat()
    return_text = Chat.generate('请你给我讲一个关于小羊的笑话。', stream=True)
    print(return_text)
