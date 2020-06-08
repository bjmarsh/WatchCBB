import urllib3

def get_html(url):
    """Use urllib3 to retrieve the html source of a given url"""

    http = urllib3.PoolManager()
    r = http.request('GET', url)
    data = r.data
    r.release_conn()
    return data
