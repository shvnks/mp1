from matplotlib import pyplot as plt
import os

# 510 files
businessList = os.listdir('mp1/BBC/business')
business = len(businessList)
# 386 files
entertainmentList = os.listdir('mp1/BBC/entertainment')
entertainment = len(entertainmentList)
# 417 files
politicList = os.listdir('mp1/BBC/politics')
politics = len(politicList)
# 511 files
sportList = os.listdir('mp1/BBC/sport')
sport = len(sportList)
# 401 files
techList = os.listdir('mp1/BBC/tech')
tech = len(techList)

# plotting chart
dev_x = ['business', 'entertainment', 'politics', 'sport', 'tech']
dev_y = [business, entertainment, politics, sport, tech]

plt.bar(dev_x, dev_y)
plt.title("Documents per section")
plt.ylabel("Number of documents")
plt.tight_layout()

plt.savefig("BBC-distribution.pdf", bbox_inches="tight")
