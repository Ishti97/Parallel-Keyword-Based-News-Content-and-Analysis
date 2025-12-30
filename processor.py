
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

# Cache for translations to avoid repeated API calls
_translation_cache = {}


def translate_to_bangla(keyword):
    """
    Translate a keyword from English to Bangla using Google Translate API.
    
    Args:
        keyword: The English keyword to translate
    
    Returns:
        List of Bangla translations
    """
    keyword_lower = keyword.lower().strip()
    
    # Check cache first
    if keyword_lower in _translation_cache:
        return _translation_cache[keyword_lower]
    
    translations = []
    
    try:
        # Get translation to Bangla
        translator = GoogleTranslator(source='en', target='bn')
        result = translator.translate(keyword_lower)
        if result:
            translations.append(result)
        
        # Cache the result
        _translation_cache[keyword_lower] = translations
        
    except Exception as e:
        print(f"      [WARN] Translation failed for '{keyword}': {type(e).__name__}")
        _translation_cache[keyword_lower] = []
    
    return translations


def get_search_keywords(keyword):
    """
    Get all search variants for a keyword (English + Bangla translations).
    
    Uses Google Translate API for dynamic translation.
    
    Args:
        keyword: The search term (in English)
    
    Returns:
        List of keywords to search for (original + translations)
    """
    keywords = [keyword.lower()]
    
    # Get Bangla translation using Google Translate API
    bangla_translations = translate_to_bangla(keyword)
    keywords.extend(bangla_translations)
    
    # Handle multi-word keywords by translating each word
    if ' ' in keyword:
        for word in keyword.split():
            if len(word) > 2:  # Skip short words like "a", "an", "the"
                word_translations = translate_to_bangla(word)
                keywords.extend(word_translations)
    
    return list(set(keywords))  # Remove duplicates


def process_page(data):
    """
    Process a fetched page and extract matching headlines.
    
    Searches for both English keyword and Bangla translations.
    
    Args:
        data: Tuple of (url, html, keyword)
    
    Returns:
        List of (url, headline_text, link) tuples
    """
    url, html, keyword = data
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    # Get all keyword variants (English + Bangla)
    search_keywords = get_search_keywords(keyword)
    
    for tag in soup.find_all("a"):
        text = tag.get_text(strip=True)
        link = tag.get("href")
        
        if text and link:
            text_lower = text.lower()
            # Check if any keyword variant matches
            for kw in search_keywords:
                if kw.lower() in text_lower or kw in text:
                    results.append((url, text, link))
                    break  # Avoid duplicate entries for same link
    
    return results
