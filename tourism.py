import kagglehub

# Download latest version
path = kagglehub.competition_download('tourism1')

print("Path to competition files:", path)