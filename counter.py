import pandas as pd

input = pd.read_csv('ground_truth_100.csv', sep = '|')
direccion = input["direccion"].notna().sum()
fot = input["fot"].notna().sum()
irregular = input["irregular"].notna().sum()
medidas = input["medidas"].notna().sum()
esquina = input["esquina"].notna().sum()
barrio = input["barrio"].notna().sum()
frentes = input["frentes"].notna().sum()
pileta = input["pileta"].notna().sum()




print(direccion, fot, irregular, medidas, esquina, barrio, frentes, pileta)


