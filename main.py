# Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Import the dataset
ds = pd.read_csv('dataset.csv')

# Display the first 5 rows of the DataFrame
# print(ds.head())

# Convert the DataFrame into a transaction format
transactions = []
for index, row in ds.iterrows():
    transaction = []
    transaction.append(f"Bedrooms_{row['Bedrooms']}")
    transaction.append(f"Bathrooms_{row['Bathrooms']}")
    transaction.append(f"Neighborhood_{row['Neighborhood']}")
    transaction.append(f"Neighborhood_{row['Neighborhood']}")
    transaction.append(f"YearBuilt_{row['YearBuilt']}")
    transaction.append(f"Price_{row['Price']}")
    transactions.append(transaction)

# Use TransactionEncoder to transform transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
ds_transformed = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to identify frequent items with a minimum support of 0.11 (adjust as needed)
frequent_itemsets = apriori(ds_transformed, min_support=0.11, use_colnames=True)

# Generate association rules from identified frequent items
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules)

# Change Labels
def map_labels(label_set):
    if 'Bathrooms_1' in label_set:
        return '1 Bathroom'
    if 'Bathrooms_2' in label_set:
        return '2 Bathrooms'
    if 'Bathrooms_3' in label_set:
        return '3 Bathrooms'
    if 'Neighborhood_Urban' in label_set:
        return 'Urban'
    if 'Neighborhood_Suburb' in label_set:
        return 'Suburban'
    if 'Neighborhood_Rural' in label_set:
        return 'Rural'
    else:
        return ', '.join(label_set)

# Apply the mapping function to convert frozenset labels to the desired string format
rules['antecedents'] = rules['antecedents'].apply(lambda x: map_labels(x))
rules['consequents'] = rules['consequents'].apply(lambda x: map_labels(x))

# Display the rules as a table
rules = rules.round(4)

plt.figure(figsize=(20, 7))
table = plt.table(cellText=rules.values,
                  colLabels=rules.columns,
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)

plt.axis('off')
plt.title('Association Rules Table')
plt.show()

# Heatmap Graph
pivot_table = rules.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='viridis', cbar=True)
plt.title('Confusion Matrix: Lift Values for Association Rules')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()


'''
# GRÁFICO GRAFO
G = nx.DiGraph()

# Adicionar arestas e nós com base nas regras de associação
for idx, rule in rules.iterrows():
    G.add_edge(rule['antecedents'], rule['consequents'], weight=rule['lift'])

# Desenhar o gráfico
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1200, edge_color='gray', linewidths=1, font_size=15)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Gráfico de Rede de Regras de Associação')
plt.xlabel('Antecedentes')
plt.ylabel('Consequentes')
plt.show()

# GRÁFICO DE DISPERSÃO
plt.figure(figsize=(12, 8))
plt.scatter(rules['confidence'], rules['lift'], s=rules['support']*1000, alpha=0.5)
plt.xlabel('Confiança (Probabilidade da Regra)')
plt.ylabel('Elevação (Lift) da Regra')
plt.title('Associação de Regras: Confiança vs. Elevação (Tamanho representa o Suporte)')
plt.grid(True)
plt.show()

# GRÁFICO DE BARRAS
# Ordenar as regras com base no suporte em ordem decrescente e selecionar as 10 principais regras
top_rules = rules.sort_values(by='support', ascending=False).head(10)

# Extrair os nomes das regras como uma lista
rule_names = [f"{antecedent} -> {consequent}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]

# Criar o gráfico de barras
plt.figure(figsize=(12, 8))
sns.barplot(x='support', y=rule_names, data=top_rules, palette='viridis')
plt.xlabel('Suporte (Frequência da Regra)')
plt.ylabel('Regras de Associação')
plt.title('Principais Regras de Associação por Suporte')
plt.show()
'''