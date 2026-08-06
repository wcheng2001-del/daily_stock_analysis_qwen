import unittest
from unittest.mock import patch

from data_provider.fundamental_adapter import AkshareFundamentalAdapter


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class EastmoneyFundamentalFallbackTestCase(unittest.TestCase):
    def test_capital_flow_uses_eastmoney_when_akshare_empty(self):
        adapter = AkshareFundamentalAdapter()

        def fake_get(url, **kwargs):
            if "fflow/kline" in url:
                return _Response({"data": {"klines": ["09:31,100,1,2,3,4"]}})
            if "daykline" in url:
                return _Response({"data": {"klines": [f"2026-08-0{i},{i},0,0,0,0" for i in range(1, 6)]}})
            return _Response(
                {
                    "data": {
                        "diff": [
                            {"f14": "半导体", "f3": 1.2, "f62": 5000},
                            {"f14": "银行", "f3": -0.8, "f62": -1000},
                        ]
                    }
                }
            )

        with patch.object(adapter, "_call_df_candidates", return_value=(None, None, [])), patch(
            "data_provider.fundamental_adapter._eastmoney_get", side_effect=fake_get
        ):
            payload = adapter.get_capital_flow("000001", top_n=1)

        self.assertEqual(payload["stock_flow"]["main_net_inflow"], 100.0)
        self.assertEqual(payload["stock_flow"]["inflow_5d"], 15.0)
        self.assertEqual(payload["sector_rankings"]["top"][0]["name"], "半导体")

    def test_dragon_tiger_uses_eastmoney_when_akshare_empty(self):
        adapter = AkshareFundamentalAdapter()

        with patch.object(adapter, "_call_df_candidates", return_value=(None, None, [])), patch(
            "data_provider.fundamental_adapter._eastmoney_datacenter",
            return_value=[{"TRADE_DATE": "2026-08-05 00:00:00"}],
        ):
            payload = adapter.get_dragon_tiger_flag("000001", lookback_days=20)

        self.assertTrue(payload["is_on_list"])
        self.assertEqual(payload["recent_count"], 1)
        self.assertEqual(payload["latest_date"], "2026-08-05")


if __name__ == "__main__":
    unittest.main()
