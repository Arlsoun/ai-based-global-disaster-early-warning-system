from src.data_loader import load_disaster_data


def test_load_disaster_data():
    df = load_disaster_data()

    assert df is not None
    assert not df.empty