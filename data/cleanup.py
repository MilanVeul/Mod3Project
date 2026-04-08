import pandas as pd

input_file = 'data/walsoorden2024-2026.csv'
output_file = 'data/walsoorden2024-2026-clean.csv'

columns = [
    'MONSTER_IDENTIFICATIE', 
    'WAARNEMINGDATUM', 
    'WAARNEMINGTIJD',
    'NUMERIEKEWAARDE'
]

try:
    df = pd.read_csv(input_file, sep=';', usecols=columns)

    # Change time format
    df['WAARNEMINGDATUM'] = pd.to_datetime(df['WAARNEMINGDATUM'], dayfirst=True).dt.strftime('%Y-%m-%d')

    df['TIMESTAMP'] = df['WAARNEMINGDATUM'] + ' ' + df['WAARNEMINGTIJD']
    # reorder & rename
    df = df[['TIMESTAMP', 'NUMERIEKEWAARDE']]
    df.columns = ['DATE TIME', 'VALUE']

    # index row
    start_value = 1052064
    df.insert(0, 'INDEX', range(start_value, start_value + len(df)))

    print(df)

    # Save to a new CSV
    df.to_csv(output_file, index=False, sep=';')
    print(f"Successfully created {output_file}")

except FileNotFoundError:
    print("Error: The input file was not found.")
except ValueError as e:
    print(f"Error: Column mismatch. {e}")