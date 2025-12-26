import re
import numpy as np
from urllib.parse import urlparse

# This function checks whether the hostname is an IP address
# Phishing URLs often use IP addresses instead of domain names
def has_ip_address(hostname: str) -> int:
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    return int(re.match(pattern, hostname) is not None)


# This function extracts numerical features from a URL string
# These features are later used by machine learning models
def extract_url_features(url: str) -> np.ndarray:
    try:
        # Parse the URL into its components (scheme, hostname, path, query, etc.)
        parsed = urlparse(url)
    except:
        # If URL parsing fails, return a zero vector
        return np.zeros(11)

    # Extract hostname and path; use empty strings if missing
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # Create a feature vector based on URL characteristics
    return np.array([
        len(url),                               # Total length of the URL
        len(hostname),                          # Length of the hostname
        len(path),                              # Length of the path
        sum(c.isdigit() for c in url),          # Number of digits in the URL
        sum(c in ['@', ':', '-', '?', '=', '%', '.', '#', '&'] for c in url),
                                                # Number of special characters
        hostname.count('.'),                    # Number of dots in the hostname
        int('@' in url),                        # Presence of '@' symbol
        int('-' in hostname),                   # Presence of '-' in hostname
        int(parsed.scheme == "https"),          # Whether HTTPS protocol is used
        has_ip_address(hostname),               # Whether hostname is an IP address
        len(parsed.query)                       # Length of query parameters
    ], dtype=float)
