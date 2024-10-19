
# %%
from utils import *

# %%
df = pd.read_excel('JCLS_DATA.xlsx')
print('no. titles:', len(df))
print('no. cols:', len(df.columns))
df.head(3)

# %%
# Data handling
# Combining different discrete proxies of the same type

discrete_proxies_raw = ['LOCUS_FANTASY', 'BFA', 'WORLD_FANTASY_AWARD', 'MYTHOPOEIC_AWARDS',
           'LOCUS_SCIFI', 'NEBULA', 'PHILIP_K_DICK_AWARD', 'J_W_CAMPBELL_AWARD', 'PROMETHEUS_AWARD', 'HUGO',
           'ROMANTIC_AWARDS', 'SCIFI_AWARDS', 'FANTASY_AWARDS', # 'HORROR_AWARDS', 
           'PULITZER', 'NBA', 'NOBEL', 
           'NORTON_ENGLISH', 'NORTON_AMERICAN', 'OPENSYLLABUS',  
           'PENGUIN_CLASSICS_SERIES_TITLEBASED', 'PENGUIN_CLASSICS_SERIES_AUTHORBASED',
           'GOODREADS_BEST_20TH_CENTURY', 'GOODREADS_CLASSICS',
           'PUBLISHERS_WEEKLY_BESTSELLERS']

# fillna to do the combinations
df[discrete_proxies_raw] = df[discrete_proxies_raw].fillna(0).astype(int)

# renaming bestsellers
df['BESTSELLERS'] = df['PUBLISHERS_WEEKLY_BESTSELLERS']
# combining Norton into one
df['NORTON'] = (df['NORTON_AMERICAN'] | df['NORTON_ENGLISH'])
# combining same-genre prizes
df['SCIFI_AWARDS'] = (df['LOCUS_SCIFI'] | df['NEBULA'] | df['PHILIP_K_DICK_AWARD'] | df['J_W_CAMPBELL_AWARD'] | df['PROMETHEUS_AWARD'] | df['HUGO']).astype(int)
df['FANTASY_AWARDS'] = (df['LOCUS_FANTASY'] | df['BFA'] | df['WORLD_FANTASY_AWARD'] | df['MYTHOPOEIC_AWARDS']).astype(int)

# And for the continuous proxies, we need to multiple the author-page rank with 100 for interpretability (very small numbers)
df['AUTH_PageRank'] = df['AUTH_PageRank'] * 100

# We end up with:
discrete_proxies = ['PENGUIN_CLASSICS_SERIES_TITLEBASED', 'PENGUIN_CLASSICS_SERIES_AUTHORBASED', 'NORTON', 'OPENSYLLABUS', 'GOODREADS_BEST_20TH_CENTURY', 'GOODREADS_CLASSICS', 
                     'NOBEL','NBA', 'PULITZER', 'SCIFI_AWARDS', 'FANTASY_AWARDS', 'ROMANTIC_AWARDS', 'BESTSELLERS']
continuous_proxies = ['AVG_RATING', 'RATING_COUNT', 'AUTH_PageRank', 'TRANSLATIONES', 'LIBRARIES', 'AUDIBLE_AVG_RATING', 'AUDIBLE_RATING_COUNT']

print('len of continuous / discrete prxies lists:', len(continuous_proxies), len(discrete_proxies))

# %%
# Create a barplot to overlook the number in each discrete proxy
labels_discrete = ['Penguin (titles)', 'Penguin (authors)', 'Norton', 'Opensyllabus', 'GR best 20th', 'GR classics', 
                    'Nobel','NBA', 'Pulitzer', 'Scifi', 'Fantasy', 'Romantic', 'Bestsellers']
numbers = {}
for i, p in enumerate(discrete_proxies):
    no = len(df.loc[df[p] == 1])
    numbers[labels_discrete[i]] = no

# Plot
r_colors = sns.color_palette("rocket", len(discrete_proxies))

plt.figure(figsize=(10,4))

sns.set_style('whitegrid')
params = {'font.family': 'serif', 'axes.labelsize': 'large', 'ytick.labelsize':'medium'}
plt.rcParams.update(params)

sns.barplot(numbers, orient='h', palette=r_colors)
plt.title('No. of titles in discrete proxies')
plt.xlabel('No. of titles')
plt.show()

# print out the numbers
print([(key, numbers[key]) for key in numbers])

# %%
# We want to correlate each of the continuous and discontinuous proxies with each other
df_filtered = df[discrete_proxies + continuous_proxies]
labels_corr = labels_discrete + ['GR avg. rating', 'GR rating count', 'WAP', 'Translations', 'Libraries', 'Audible avg. rating', 'Audible rating count']
df_filtered.columns = labels_corr

# Clustermap
corr_df = round(df_filtered.corr(method='spearman'),2)

plt.figure(figsize=(13,9))
sns.clustermap(corr_df, annot=True, method='ward', cbar_pos=None)
plt.show()

# Clustermap, setting correlations with p>0.05 to 0
print('Clustermap, setting correlations with p>0.05 to 0')
pvals = df_filtered.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*df_filtered.corr().shape)
psig = 0.05

df_corr = df_filtered.corr(method= 'spearman')[pvals<psig].fillna(0)
plt.figure(figsize=(16,15))
sns.clustermap(df_corr, annot=True,cbar_pos=None, method='ward')
plt.show()


# %%
# How linear are correlations? Check on libraries vs. avg. rating

# Let's first filter avg. rating so we get books with at least 5 ratings (for visibility)
df = df[df['RATING_COUNT'] >= 5]

# add categories
for i, row in df.iterrows():
    for j, proxy in enumerate(discrete_proxies):
        if row[proxy] > 0:
           df.at[i, 'CATEGORY'] = labels_discrete[j]

df_cat_scatter = df.loc[df['CATEGORY'].notnull()]

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
#sns.scatterplot(df, y='LIBRARIES', x='AVG_RATING', alpha=0.5, color='steelblue', size=10)
sns.scatterplot(df, y='LIBRARIES', x='AVG_RATING', alpha=0.5, color='steelblue', size=10, ax=axes[0])
sns.scatterplot(df_cat_scatter, y='LIBRARIES', x='AVG_RATING', alpha=0.5, hue='CATEGORY', ax=axes[1])


# %%
# Making a NETWORK over the correlations
dt = df[['PULITZER', 'NBA', 'NOBEL',
       'GOODREADS_BEST_20TH_CENTURY',
       'GOODREADS_CLASSICS', 'OPENSYLLABUS', 'ROMANTIC_AWARDS', 'SCIFI_AWARDS',
       'FANTASY_AWARDS', 'PENGUIN_CLASSICS_SERIES_AUTHORBASED',
       'TRANSLATIONES', 'AUTH_PageRank', 'NORTON']].copy()
# Renaming columns for layout
dt.columns = ['Pulitzer', 'NBA', 'Nobel', 'GR best 20th century',
       'GR classics', 'Opensyllabus', 'Romantic awards', 'Scifi awards',
       'Fantasy awards', 'Penguin classics', 'Translations', 'WAP',
       'Norton anthology'] # note, we're only using the author-based penguin classics

# We need to make some columns with continuous values into binaries (we only want to know if a title exists within a group, not how many)
dt['Translations']= dt.Translations.apply(lambda x: 1 if x != 0 else 0)
dt['WAP']= dt.WAP.apply(lambda x: 1 if x != 0 else 0)

# Do the correlation matrix
corr_df = dt.corr(method='spearman')

## We make the EDGES df
edges = corr_df.stack().reset_index()
edges.columns = ['source', 'target', 'weight']
# introducing a THRESHOLD in the correlation strength so we don't plot all connections
threshold = 0.05
edges = edges[edges["weight"] > threshold]

## Get a list of names
names = corr_df.columns

## We have to remove duplicates (e.g. "Pulitzer - NBA" & "NBA - Pulitzer") as well as self-correlations (pulitzer-pulitzer = 1)
# Remove duplicates based on two columns
swap = edges['source'] < edges['target']
edges.loc[swap, ['source', 'target']] = edges.loc[swap, ['target', 'source']].values
edges = edges.drop_duplicates(subset=['source', 'target'], keep='first')
# Remove self-correlations
edges = edges.loc[edges['weight'] != 1]

# Just check one of them
print('printout, seeing the correlations')
print(edges.loc[edges['target']=='Fantasy awards'])

# ## Create the NODES df
# # We make a distance matrix out of the correlation meatrix
distance = 1-corr_df
# # We do a dimension reduction so we can get coordinates
dim_red = TSNE(n_components = 2, metric="precomputed", perplexity=10, init="random")
# # And get the coordinates
x, y = dim_red.fit_transform(distance).T
# # And we compute the sizes of the nodes depending on number of titles in each standard-list
sizes = dt[names].sum(axis=0).astype(int).loc[names].to_numpy()
# # Create the nodes df
nodes = pd.DataFrame({'names':names, 'sizes':sizes, 'x': x, 'y': y})
# # Add color
nodes['color'] = ['#1f77b4', '#1f77b4', '#1f77b4', '#7f7f7f', '#7f7f7f', '#7f7f7f', '#2ca02c','#2ca02c', '#2ca02c', '#17becf', '#7f7f7f', '#7f7f7f', '#17becf' ]

# we don't use our own coordinates but determine position by using the Fruchterman-Reingold force-directed algorithm : nx.spring_layout()
# - edge width and opacity are determined by the correlation strength (weight), 
# - node size is determined by group/list-size
# note that when we use this method, the graph will look different every time we run the code

# Create graph object
G = nx.Graph()

# Add nodes with attributes
for i, row in nodes.iterrows():
    G.add_node(row['names'], size=row['sizes'], color=row['color'])

# Add edges with weights
for i, row in edges.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# Calculate node positions using spring layout
pos = nx.spring_layout(G, k=0.5)

# Get edge weights as a dictionary
weights = nx.get_edge_attributes(G, 'weight')

# Define edge thickness and opacity based on weights
widths = [weights[edge]*9 for edge in G.edges()]
opacities = [weights[edge]*1.5 for edge in G.edges()]

# Create a new figure with a bigger size
fig, ax = plt.subplots(figsize=(10, 10))

# Draw edges with thickness and opacity based on weights
nx.draw_networkx_edges(G, pos, width=widths, alpha=opacities, edge_color='gray')

# Draw nodes with size as node size and color as node color
node_sizes = nodes['sizes'] * 5  # Scale up node sizes for visibility
node_colors = nodes['color']
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.18)

# Label nodes with names
node_labels = {}
for i, row in nodes.iterrows():
    node_labels[row['names']] = row['names']
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family="serif")

# Display the plot
plt.axis('off')
plt.show()


# %%
# we want to note the averages of continuous measures for each discrete measure

averages = {}
for cp in continuous_proxies:
    dict_means = {} 
    dat = df.loc[df[cp].notnull()]
    dat = dat.loc[dat[cp] != 0]
    #dat = dat.loc[dat[cp] != 'x']
    for proxy in discrete_proxies:
        #mean_, std_ = dat[cp].loc[dat[proxy] == 1].mean(), dat[cp].loc[dat[proxy] == 1].std()
        proxy_df = dat.loc[dat[proxy] == 1]
        mean_, std_ = proxy_df[cp].mean(), proxy_df[cp].std()

        if cp == 'AUTH_PageRank':
            dict_means[proxy] = (round(mean_,6), round(std_,6))
        else:
            dict_means[proxy] = (round(mean_,2), round(std_,2))
    averages[cp] = dict_means

averages_df = pd.DataFrame(averages)
averages_df.head(15)

# %%
# We want to see the distribution of each continuous proxy for each discrete proxy

# function to plot various kdeplots
def plot_kdeplots(df, cont_proxies, plts_per_row, l, h):
    plots_per_row = plts_per_row

    if len(cont_proxies) <= plots_per_row:
        fig, axes_list = plt.subplots(1, len(cont_proxies), figsize=(l, h), dpi=300)#, sharey=True)
    else:
        rows = len(cont_proxies) // plots_per_row
        if len(cont_proxies) % plots_per_row != 0:
            rows += 1
        fig, axes_list = plt.subplots(rows, plots_per_row, figsize=(l, h * rows), dpi=300)#, sharey=True)
        
    fig.tight_layout(pad=3)

    labels_kdeplot = [x for x in cont_proxies]

    for i, score in enumerate(cont_proxies):
        plt.tight_layout()

        sns.set(style="whitegrid", font_scale=2, font='serif')

        ax = axes_list.flat[i]

        df[score] = df[score].astype("float64")

        #sns.kdeplot(data=df[score], ax=ax, hue='CATEGORY')
        sns.kdeplot(df.reset_index(drop=True), x=score, hue='CATEGORY', ax=ax, log_scale=False, linewidth=4, palette='tab10', alpha=0.5, 
                    warn_singular=False) # remove warning on variance == 0 --> due to np.isclose being used, see: https://github.com/mwaskom/seaborn/issues/2294

        # Set labels
        ax.set_xlabel(labels_kdeplot[i])
        ax.set_xlim(0, df[score].max())
        
        #if i >= 1:
            #ax.set_ylabel('')  # Set the y-axis label to an empty string
            #ax.legend([],[], frameon=False)
    
    # remove last empty plot
    axes_list.flat[-1].set_visible(False)
            
    plt.show()
    return fig


# add discrete proxy categories to our titles
for i, row in df.iterrows():
    for j, proxy in enumerate(discrete_proxies):
        if row[proxy] > 0:
           df.at[i, 'CATEGORY'] = labels_discrete[j]

# remove all titles that are not in a discrete category
df_kdeplots = df.loc[df['CATEGORY'].notnull()]

# and remove the very very high outliers in rating count
df_kdeplots = df_kdeplots.loc[df_kdeplots['RATING_COUNT'] <= 100000]

# remove audible continuous category, since it has only 600 titles
continuous_proxies_minus = ['AVG_RATING', 'RATING_COUNT', 'AUTH_PageRank', 'TRANSLATIONES', 'LIBRARIES']

# plot it
x = plot_kdeplots(df_kdeplots, continuous_proxies_minus, 2, 35, 8)


# %%

# make dfs for each discrete proxy
proxy_dfs = []
for proxy in discrete_proxies:
    proxy_df = df.loc[df[proxy] == 1]
    proxy_dfs.append(proxy_df)

for cp in continuous_proxies_minus:

    # we get corpus average
    corpus_average = df[cp].loc[(df[cp].notnull())].mean()

    data = []
    for dataf in proxy_dfs:
        data.append(dataf[cp])

    fig = plt.figure(figsize =(30, 13)) 
    plt.style.use('ggplot')
    plt.rcParams["font.family"] = "serif"

    # Creating plot 
    plt.boxplot(data, labels=labels_discrete, sym='.') 

    plt.ylabel(cp, fontsize=22)

    # Let's also add a line that indicates the corpus average library holding
    plt.axhline(corpus_average, ls='-', c='grey', lw=1)

    plt.xticks(rotation=30, ha='center')

    # show plot 
    plt.show()


# %%
print('All done!')
# %%
