import json
import boto3
import json_repair
from termcolor import colored
from lib import frames

MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'
MODEL_VER = 'bedrock-2023-05-31'
CLAUDE_PRICING = (0.00025, 0.00125)


def analyze_conversations(transcript_file):

    messages = []

    # transcript
    transcript_message = make_transcript(transcript_file)
    messages.append(transcript_message)

    # output format?
    messages.append({
        'role': 'assistant',
        'content': 'Got the transcript. What output format?'
    })

    # example output
    example_message = make_conversation_example()
    messages.append(example_message)

    # prefill output
    messages.append({
        'role': 'assistant',
        'content': '{'
    })

    ## system prompt to role play
    system = 'You are a media operation assistant who analyses movie transcripts in WebVTT format and suggest chapter points based on the topic changes in the conversations. It is important to read the entire transcripts.'

    ## setting up the model params
    model_params = {
        'anthropic_version': MODEL_VER,
        'max_tokens': 4096,
        'temperature': 0.1,
        'top_p': 0.7,
        'top_k': 20,
        'stop_sequences': ['\n\nHuman:'],
        'system': system,
        'messages': messages
    }

    try:
        response = inference(model_params)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(model_params)
    return response

def get_contextual_information(images, text):
    task_all = 'You are asked to provide the following information: a detail description to describe the scene, sentiment, and brands and logos that may appear in the scene, and five most relevant tags from the scene.'
    system = 'You are a media operation engineer. Your job is to review a portion of a video content presented in a sequence of consecutive images. Each image also contains a sequence of frames presented in a 4x7 grid reading from left to right and then from top to bottom. You may also optionally be given the conversation of the scene that helps you to understand the context. {0} It is important to return the results in JSON format and also includes a confidence score from 0 to 100. Skip any explanation.';

    messages = []
 
    # adding sequences of composite images to the prompt
    message_images = make_image_message(images)
    messages.append(message_images)

    # adding the conversation to the prompt
    messages.append({
        'role': 'assistant',
        'content': 'Got the images. Do you have the conversation of the scene?'
    })
    message_conversation = make_conversation_message(text)
    messages.append(message_conversation)

    # other information
    messages.append({
        'role': 'assistant',
        'content': 'OK. Do you have other information to provdie?'
    })

    other_information = []

    ## Sentiment
    sentiment_list = make_sentiments()
    other_information.append(sentiment_list)

    messages.append({
        'role': 'user',
        'content': other_information
    })

    # output format
    messages.append({
        'role': 'assistant',
        'content': 'OK. What output format?'
    })
    output_format = make_output_example()
    messages.append(output_format)

    # prefill '{'
    messages.append({
        'role': 'assistant',
        'content': '{'
    })
    
    model_params = {
        'anthropic_version': MODEL_VER,
        'max_tokens': 4096,
        'temperature': 0.1,
        'top_p': 0.7,
        'top_k': 20,
        'stop_sequences': ['\n\nHuman:'],
        'system': system.format(task_all),
        'messages': messages
    }

    try:
        response = inference(model_params)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(model_params)

    return response

def inference(model_params):
    model_id = MODEL_ID
    accept = 'application/json'
    content_type = 'application/json'

    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

    response = bedrock_runtime_client.invoke_model(
        body=json.dumps(model_params),
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    response_body = json.loads(response.get('body').read())

    # patch the json string output with '{' and parse it
    response_content = response_body['content'][0]['text']
    if response_content[0] != '{':
        response_content = '{' + response_content

    try:
        response_content = json.loads(response_content)
    except Exception as e:
        print(colored("Malformed JSON response. Try to repair it...", 'red'))
        try:
            response_content = json_repair.loads(response_content)
        except Exception as e:
            print(colored("Failed to repair the JSON response...", 'red'))
            print(colored(response_content, 'red'))
            raise e

    response_body['content'][0]['json'] = response_content

    return response_body

def display_conversation_cost(response, display=True):
    # us-east-1 pricing
    input_per_1k, output_per_1k = CLAUDE_PRICING

    input_tokens = response['usage']['input_tokens']
    output_tokens = response['usage']['output_tokens']

    conversation_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    if display:
        print('\n')
        print('========================================================================')
        print('Estimated cost:', colored(f"${conversation_cost}", 'green'), f"in us-east-1 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
        print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': conversation_cost,
    }

def display_contextual_cost(usage, display=True):
    # us-east-1 pricing
    input_per_1k, output_per_1k = CLAUDE_PRICING

    input_tokens = usage['input_tokens']
    output_tokens = usage['output_tokens']

    contextual_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    if display:
        print('\n')
        print('========================================================================')
        print('Estimated cost:', colored(f"${round(contextual_cost, 4)}", 'green'), f"in us-east-1 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
        print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': contextual_cost,
    }

def make_conversation_example():
    example = {
        'chapters': [
            {
                'start': '00:00:10.000',
                'end': '00:00:32.000',
                'reason': 'It appears the chapter talks about...'
            }
        ]
    }

    return {
        'role': 'user',
        'content': 'JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))
    }

def make_transcript(transcript_file):
    with open(transcript_file, encoding="utf-8") as f:
        transcript = f.read()
    
    return {
        'role': 'user',
        'content': 'Here is the transcripts in <transcript> tag:\n<transcript>{0}\n</transcript>\n'.format(transcript)
    }

def make_image_message(images):
    # adding the composite image sequences
    image_contents = [{
        'type': 'text',
        'text': 'Here are {0} images containing frame sequence that describes a scene.'.format(len(images))
    }]

    for image in images:
        bas64_image = frames.image_to_base64(image)
        image_contents.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/jpeg',
                'data': bas64_image
            }
        })

    return {
        'role': 'user',
        'content': image_contents
    }

def make_conversation_message(text):
    message = {
        'role': 'user',
        'content': 'No conversation.'
    }

    if text:
        message['content'] = 'Here is the conversation of the scene in <conversation> tag.\n<conversation>\n{0}\n</conversation>\n'.format(text)

    return message

def make_sentiments():
    sentiments = ['Positive', 'Neutral', 'Negative', 'None']

    return {
        'type': 'text',
        'text': 'Here is a list of Sentiments in <sentiment> tag:\n<sentiment>\n{0}\n</sentiment>\nOnly answer the sentiment from this list.'.format('\n'.join(sentiments))
    }

def make_output_example():
    example = {
        'description': {
            'text': 'The scene describes...',
            'score': 98
        },
        'sentiment': {
            'text': 'Positive',
            'score': 90
        },
        'brands_and_logos': [
            {
                'text': 'Amazon',
                'score': 95
            },
            {
                'text': 'Nike',
                'score': 85
            }
        ],
        'relevant_tags': [
            {
                'text': 'BMW',
                'score': 95
            }
        ]            
    }
    
    return {
        'role': 'user',
        'content': 'Return JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))
    }