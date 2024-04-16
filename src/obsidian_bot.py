import langchain as lc
from langchain.chains import OpenAIChain
from langchain.schema import LLMConfig
from langchain_community.document_loaders import ObsidianLoader

# https://python.langchain.com/docs/integrations/document_loaders/obsidian/
# Path to your Obsidian Vault
OBSIDIAN_VAULT_PATH = 'path/to/your/obsidian/vault'

def load_obsidian_data(vault_path):
    """
    Load data from Obsidian vault.
    This function should be customized to parse the markdown files or other data formats
    you have in your Obsidian vault.
    """
    # Example: Load all markdown files as a single string of text
    # You might want to process these files to extract structured data or important segments
    #data = load_text(vault_path, extension="md")
    loader = ObsidianLoader(vault_path)
    data = loader.load()
    return data

def setup_langchain(data):
    """
    Setup LangChain with the loaded Obsidian data.
    Here we use a simple configuration with OpenAI's language models.
    Adjust the configuration as needed.
    """
    # Configuration for the language model (use your OpenAI API key)
    llm_config = LLMConfig(
        model="text-davinci-002",  # Choose the appropriate model
        api_key="your-openai-api-key"
    )
    
    # Initialize LangChain with a simple text-based retriever
    chain = OpenAIChain.from_text(
        text=data,
        llm_config=llm_config
    )
    return chain

def main():
    print("Loading data from Obsidian Vault...")
    data = load_obsidian_data(OBSIDIAN_VAULT_PATH)
    
    print("Setting up LangChain...")
    chain = setup_langchain(data)
    
    # Example query
    query = "What is the main idea of linked thinking?"
    print(f"Querying LangChain: {query}")
    response = chain.query(query)
    print("Response from LangChain:", response)

if __name__ == "__main__":
    main()
