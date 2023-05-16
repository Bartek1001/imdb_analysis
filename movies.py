import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.style.use('ggplot')
#from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12,8)

#read data
df = pd.read_csv('movies.csv')

#data type, missing data
print(df.info())

df['budget'] = df['budget'].fillna(0)
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].fillna(0)
df['gross'] = df['gross'].astype('int64')

pd.set_option('display.max_rows',None)
df.sort_values(by=['gross'], ascending=False)

#drop any duplicates/distinct values
df['company'].drop_duplicates().sort_values(ascending=False)


plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Movie budget vs Gross earnings')
plt.xlabel('movie budget')
plt.ylabel('gross earnings')
plt.savefig("Movie budget vs Gross earnings")
plt.show()

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})
plt.title('Linear regression')
plt.savefig("Linear regression")
plt.show()


plt.title('IMDB user movie rating')
plt.xlabel('rating')
plt.ylabel('quantity')
plt.hist(df['score'])
plt.savefig('IMDB user movie rating')
plt.show()


df['director'] = df['director'].fillna(' ')
df['director'] = df['director'].astype('string')
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(background_color='White', width=1920,height=1080).generate(" ".join(df['director']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Director of the movie Word Cloud')
plt.savefig('Director of the movie Word Cloud')
plt.show()


df['star'] = df['star'].fillna(' ')
df['star'] = df['star'].astype('string')
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(background_color='Black', width=1920,height=1080).generate(" ".join(df['star']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Movie star Word Cloud')
plt.savefig('Movie star Word Cloud')
plt.show()

df['writer'] = df['writer'].fillna(' ')
df['writer'] = df['writer'].astype('string')
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(background_color='White', width=1920,height=1080).generate(" ".join(df['writer']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Writer of the movie Word Cloud')
plt.savefig('Writer of the movie Word Cloud')
plt.show()


country = df['country'].value_counts()
country = pd.DataFrame(country)
country = country.head(10)

sns.barplot(x = country.index, y = country['country'])

labels = country.index.tolist()
plt.gcf().set_size_inches(10, 5)

plt.title('Country vs Movies released')
plt.xlabel('country')
plt.ylabel('movies released')
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9] , labels = labels, rotation = 30)
plt.savefig('Country vs Movies released')
plt.show()


company = df['company'].value_counts()
company = pd.DataFrame(company)
company = company.head(10)

sns.barplot(x = company.index, y = company['company'])

labels = company.index.tolist()
plt.gcf().set_size_inches(10, 5)

plt.title('Company vs Movies released')
plt.xlabel('company')
plt.ylabel('movies released')
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9] , labels = labels, rotation = 30)
plt.savefig('Company vs Movies released')
plt.show()

genre = df['genre'].value_counts()
genre = pd.DataFrame(genre)
s = sum(genre['genre'][5:])
genre = genre.head(5)
genre.loc["Other genres"] = s

genre.plot.pie(autopct='%1.1f%%',subplots=True,figsize=(10,8))
plt.title('Genres', fontsize = 20)
plt.tight_layout()
plt.savefig('Genres')
plt.show()


print(df.corr())  ##method = 'pearson'/'kendall'/'spearman'

correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matric for Numeric Movie Features')
plt.savefig('Correlation Matric for Numeric Movie Features')
plt.show()

df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matric for all Movie Features')
plt.savefig('Correlation Matric for all Movie Features')
plt.show()

correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
print(corr_pairs)

sorted_pairs = corr_pairs.sort_values()
print(sorted_pairs)