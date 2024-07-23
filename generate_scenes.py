import openai

def get_memorable_scenes(movie_title, api_key):
    """
    Generates 10 memorable scenes from a given movie using ChatGPT.

    Args:
        movie_title: The title of the movie.
        api_key: Your OpenAI API key.

    Returns:
        A list of 10 memorable scenes.
    """
    openai.api_key = api_key

    prompt = f"Generate the 10 most memorable scenes from the movie '{movie_title}' in a numbered list."

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify the GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    scenes_text = response['choices'][0]['message']['content'].strip()

    # Splitting by the numbered list format
    scenes = []
    for line in scenes_text.split('\n'):
        if line.strip():
            if line[0].isdigit() and line[1] in ['.', ')']:
                scenes.append(line.strip())
            elif scenes:
                scenes[-1] += ' ' + line.strip()

    return scenes

if __name__ == "__main__":
    api_key = ""
    movie_title = input("Enter the movie title: ")

    scenes = get_memorable_scenes(movie_title, api_key)

    for i, scene in enumerate(scenes, start=1):
        print(f"Scene {scene}")
