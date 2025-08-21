import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import emoji
from collections import Counter
import os

# Function to parse WhatsApp chat log
def parse_whatsapp_chat(file_path):
    """
    Parse a WhatsApp chat .txt file into a structured format.
    Args:
        file_path (str): Path to the exported WhatsApp chat file.
    Returns:
        list: List of dictionaries containing date, author, and message.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Chat file {file_path} not found.")
    except UnicodeDecodeError:
        raise UnicodeDecodeError("File encoding issue. Ensure the file is UTF-8 encoded.")

    # Regex to match WhatsApp chat format (e.g., "MM/DD/YY, HH:MM - Author: Message")
    pattern = r'(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}\s*[AP]M) - ([^:]+): (.+)'
    chat_data = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            date, author, message = match.groups()
            chat_data.append({'Date': date, 'Author': author.strip(), 'Message': message.strip()})
        else:
            # Handle multi-line messages
            if chat_data and line.strip():
                chat_data[-1]['Message'] += ' ' + line.strip()

    return chat_data

# Function to extract emojis
def extract_emojis(text):
    """
    Extract all emojis from a text string.
    Args:
        text (str): Input text containing potential emojis.
    Returns:
        list: List of emojis found in the text.
    """
    return [char for char in text if char in emoji.UNICODE_EMOJI['en']]

# Function to extract URLs
def extract_urls(text):
    """
    Extract URLs from a text string.
    Args:
        text (str): Input text containing potential URLs.
    Returns:
        list: List of URLs found in the text.
    """
    url_pattern = r'(https?://[^\s]+)'
    return re.findall(url_pattern, text)

# Function to generate word cloud
def generate_wordcloud(text, title, filename):
    """
    Generate and save a word cloud from text.
    Args:
        text (str): Input text for word cloud.
        title (str): Title for the plot.
        filename (str): Output filename for the word cloud image.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Main analysis function
def analyze_whatsapp_chat(file_path, output_dir='output'):
    """
    Perform WhatsApp chat analysis and generate visualizations.
    Args:
        file_path (str): Path to the WhatsApp chat file.
        output_dir (str): Directory to save output files.
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse chat data
    chat_data = parse_whatsapp_chat(file_path)
    df = pd.DataFrame(chat_data)

    # Handle missing values
    df['Author'] = df['Author'].fillna('Unknown')
    df['Message'] = df['Message'].fillna('')

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y, %I:%M %p', errors='coerce')

    # Calculate basic statistics
    total_messages = df.shape[0]
    media_messages = df[df['Message'] == '<Media omitted>'].shape[0]

    # Extract emojis
    df['Emojis'] = df['Message'].apply(extract_emojis)
    all_emojis = [emoji for sublist in df['Emojis'] for emoji in sublist]
    emoji_counts = Counter(all_emojis)
    total_emojis = len(all_emojis)

    # Extract URLs
    df['URLs'] = df['Message'].apply(extract_urls)
    total_urls = len([url for sublist in df['URLs'] for url in sublist])

    # Per-author statistics
    author_stats = {}
    for author in df['Author'].unique():
        if author != 'Unknown':
            author_df = df[df['Author'] == author]
            messages_sent = author_df.shape[0]
            media_sent = author_df[author_df['Message'] == '<Media omitted>'].shape[0]
            emojis_sent = len([emoji for sublist in author_df['Emojis'] for emoji in sublist])
            urls_sent = len([url for sublist in author_df['URLs'] for url in sublist])
            avg_words = np.mean([len(msg.split()) for msg in author_df['Message'] if msg != '<Media omitted>'])
            author_stats[author] = {
                'Messages Sent': messages_sent,
                'Media Sent': media_sent,
                'Emojis Sent': emojis_sent,
                'URLs Sent': urls_sent,
                'Avg Words per Message': round(avg_words, 2)
            }

    # Save processed data
    df.to_csv(os.path.join(output_dir, 'processed_chat_data.csv'), index=False)

    # Generate visualizations
    # Emoji distribution
    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['Emoji', 'Count'])
    if not emoji_df.empty:
        fig = px.bar(emoji_df.head(10), x='Emoji', y='Count', title='Top 10 Emojis Used')
        fig.update_layout(xaxis_title="Emoji", yaxis_title="Frequency")
        fig.update_xaxes(tickangle=45)
        fig.update_traces(text=emoji_df['Count'].head(10), textposition='auto')
        fig.update_layout(showlegend=False)
        fig.update_layout(template='plotly_white')
        fig.update_layout(title_x=0.5)
        fig.update_yaxes(showgrid=True)
        fig.update_xaxes(showgrid=False)
        fig.update_layout(margin=dict(l=50, r=50, t=80, b=50))
        fig.update_layout(font=dict(family="Arial", size=12))
        fig.update_layout(height=400)
        fig.update_layout(width=800)
        fig.update_traces(marker_color='#636EFA')
        fig.update_layout(hovermode='closest')
        fig.update_layout(xaxis=dict(showline=True, linewidth=2, linecolor='black'))
        fig.update_layout(yaxis=dict(showline=True, linewidth=2, linecolor='black'))
        fig.update_layout(title_font=dict(size=20, family="Arial", color="black"))
        fig.update_layout(xaxis_title_font=dict(size=14, family="Arial", color="black"))
        fig.update_layout(yaxis_title_font=dict(size=14, family="Arial", color="black"))
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_traces(textfont=dict(size=12, family="Arial", color="black"))
        fig.update_traces(hovertemplate='Emoji: %{x}<br>Count: %{y}<extra></extra>')
        fig.update_layout(autosize=False)
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"))
        fig.update_layout(bargap=0.2)
        fig.update_layout(bargroupgap=0.1)
        fig.update_layout(separators='.,')
        fig.update_layout(xaxis_tickfont=dict(size=12, family="Arial", color="black"))
        fig.update_layout(yaxis_tickfont=dict(size=12, family="Arial", color="black"))
        fig.update_layout(clickmode='event+select')
        fig.update_layout(dragmode='zoom')
        fig.update_layout(hoverdistance=10)
        fig.update_layout(spikedistance=10)
        fig.update_layout(xaxis_zeroline=False)
        fig.update_layout(yaxis_zeroline=False)
        fig.update_layout(xaxis_showspikes=True)
        fig.update_layout(yaxis_showspikes=True)
        fig.update_layout(xaxis_spikecolor="black")
        fig.update_layout(yaxis_spikecolor="black")
        fig.update_layout(xaxis_spikesnap="cursor")
        fig.update_layout(yaxis_spikesnap="cursor")
        fig.update_layout(xaxis_spikethickness=1)
        fig.update_layout(yaxis_spikethickness=1)
        fig.update_layout(xaxis_spikedash='solid')
        fig.update_layout(yaxis_spikedash='solid')
        fig.update_layout(xaxis_spikemode='across')
        fig.update_layout(yaxis_spikemode='across')
        fig.update_layout(xaxis_range=[-0.5, len(emoji_df.head(10))-0.5])
        fig.update_layout(yaxis_range=[0, emoji_df['Count'].head(10).max() * 1.1])
        fig.update_layout(xaxis_constrain='domain')
        fig.update_layout(yaxis_constrain='domain')
        fig.update_layout(xaxis_autorange=False)
        fig.update_layout(yaxis_autorange=False)
        fig.update_layout(xaxis_automargin=True)
        fig.update_layout(yaxis_automargin=True)
        fig.update_layout(xaxis_title_standoff=15)
        fig.update_layout(yaxis_title_standoff=15)
        fig.update_layout(xaxis_ticklen=5)
        fig.update_layout(yaxis_ticklen=5)
        fig.update_layout(xaxis_tickwidth=1)
        fig.update_layout(yaxis_tickwidth=1)
        fig.update_layout(xaxis_tickcolor='black')
        fig.update_layout(yaxis_tickcolor='black')
        fig.update_layout(xaxis_gridcolor='lightgrey')
        fig.update_layout(yaxis_gridcolor='lightgrey')
        fig.update_layout(xaxis_zerolinecolor='black')
        fig.update_layout(yaxis_zerolinecolor='black')
        fig.update_layout(xaxis_zerolinewidth=1)
        fig.update_layout(yaxis_zerolinewidth=1)
        fig.update_layout(xaxis_mirror=True)
        fig.update_layout(yaxis_mirror=True)
        fig.update_layout(xaxis_showline=True)
        fig.update_layout(yaxis_showline=True)
        fig.update_layout(xaxis_linecolor='black')
        fig.update_layout(yaxis_linecolor='black')
        fig.update_layout(xaxis_linewidth=2)
        fig.update_layout(yaxis_linewidth=2)
        fig.write_html(os.path.join(output_dir, 'emoji_distribution.html'))

    # Word clouds
    all_text = ' '.join(df['Message'][df['Message'] != '<Media omitted>'])
    generate_wordcloud(all_text, 'Overall Word Cloud', os.path.join(output_dir, 'overall_wordcloud.png'))

    for author in author_stats.keys():
        author_text = ' '.join(df[df['Author'] == author]['Message'][df['Message'] != '<Media omitted>'])
        generate_wordcloud(author_text, f'Word Cloud for {author}', os.path.join(output_dir, f'{author}_wordcloud.png'))

    # Print summary
    print(f"Total Messages: {total_messages}")
    print(f"Media Shared: {media_messages}")
    print(f"Emojis Shared: {total_emojis}")
    print(f"Links Shared: {total_urls}")
    print("\nPer-Author Stats:")
    for author, stats in author_stats.items():
        print(f"\n{author}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

# Example usage
# analyze_whatsapp_chat('your_chat.txt', 'output')