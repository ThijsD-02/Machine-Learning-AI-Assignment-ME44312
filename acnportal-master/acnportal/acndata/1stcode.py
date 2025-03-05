from data_client import DataClient

client = DataClient(api_token="bsVJ2JK--lixPnRZYLn4OXPgYxd0ISUXOcZ3ZsAK1kE")

print(client.get_sessions(site="caltech"))


