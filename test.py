test_csv = r"C:\Users\kprajapati\Documents\Learn\assgn\George\Dataset2\12th August 2021\Participant1\leftswipe\20210812-0013_01.csv"
df = pd.read_csv(test_csv, header=0, skip_blank_lines=True)
# dropping first row as it shows unit
df.drop([df.index[0]], inplace=True)

df = df[3000:8000]
# converting everything from string to float
df = df.applymap(lambda x : float(x))

# compute features
df['EWM_mean'] = df.iloc[:,1].ewm(span=200,adjust=True).mean()
df.drop(['Time', 'Channel A'], inplace=True, axis=1)


nw_df = df.T
# nw_df.head()

test_tensor = torch.tensor(nw_df.loc['EWM_mean',:].values, dtype=torch.float32).unsqueeze_(dim=0)
test_loader = torch.unsqueeze(test_tensor, dim=0)

model = torch.load('model.pt')
model.eval()

op = model(test_loader)
predicted_classes = torch.max(op, 1)[1]

print(predicted_classes.item())

lb = np.load('label_encoder.npy')
predict = lb.inverse_transform(predicted_classes)
print(predict[0])
