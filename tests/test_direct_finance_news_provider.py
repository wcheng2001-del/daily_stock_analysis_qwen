import unittest
from unittest.mock import patch

from src.search_service import DirectFinanceNewsProvider


class _Response:
    def __init__(self, *, content=b"", json_payload=None):
        self.content = content
        self._json_payload = json_payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_payload


class DirectFinanceNewsProviderTestCase(unittest.TestCase):
    def test_yahoo_rss_is_used_for_us_symbols(self):
        provider = DirectFinanceNewsProvider()
        rss = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss><channel><item>
          <title>NVDA rises after earnings</title>
          <description>Nvidia shares moved higher.</description>
          <link>https://finance.yahoo.com/news/nvda</link>
          <pubDate>Thu, 06 Aug 2026 10:00:00 GMT</pubDate>
        </item></channel></rss>"""

        with patch("src.search_service.requests.get") as mock_get, patch.object(
            DirectFinanceNewsProvider, "_fetch_global_fast_news", return_value=[]
        ):
            mock_get.return_value = _Response(content=rss)
            response = provider.search("NVIDIA NVDA stock latest news", max_results=3)

        self.assertTrue(response.success)
        self.assertEqual(response.provider, "DirectFinanceNews")
        self.assertEqual(response.results[0].source, "Yahoo Finance")
        self.assertIn("NVDA", response.results[0].title)


if __name__ == "__main__":
    unittest.main()
