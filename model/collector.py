import configparser

config = configparser.ConfigParser()
config.read("config.ini")

config_path = config.get('PATH', 'QUERIES')


def collect_queries() -> list[str]:
    queries = []

    with open(config_path, 'r') as file:
        for line in file.readlines():
            if line.strip():
                line = line.strip().replace('  ', ' ').replace(' ', '+')
                queries.append(line)

    return queries

