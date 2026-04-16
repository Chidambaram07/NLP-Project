# Reddit Pain Points Scraper

**Scrape Reddit complaints → Detect pain points → Find business opportunities**

A production-ready system to scrape Reddit posts, detect pain points and opportunities, store in MongoDB Atlas, and expose via FastAPI.

---

## **What This Project Does**

1.  Scrapes 30+ subreddits for complaints and pain points
2.  Detects pain points using keyword analysis
3.  Categorizes problems (Career, Finance, Health, etc.)
4.  Stores data in MongoDB Atlas (cloud database)
5.  Exposes REST APIs for teammates
6.  Identifies business opportunities from pain points

---

## **Project Structure**

```
reddit-pain-points-scraper/
├── config/              # Settings & database
├── scraper/             # Reddit scraping logic
├── database/            # MongoDB operations
├── api/                 # FastAPI endpoints
├── scripts/             # Run scraper manually
└── .env                 # Credentials (YOU MUST CREATE THIS)
```

---

## **Setup Instructions**

### **Step 1: Install Python Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 2: Get Reddit API Credentials**

1. Go to: https://www.reddit.com/prefs/apps
2. Click **"Create App"** or **"Create Another App"**
3. Fill form:
   - **Name:** Reddit Pain Points Scraper
   - **App type:** Select **"script"**
   - **Description:** Pain point detection
   - **Redirect URI:** http://localhost:8080
4. Click **"Create app"**
5. Copy these values:
   - **Client ID** (under app name, ~14 characters)
   - **Client Secret** (shown as "secret")

### **Step 3: Setup MongoDB Atlas (Cloud Database)**

**3.1 Create MongoDB Account**
1. Go to: https://www.mongodb.com/cloud/atlas/register
2. Sign up (free tier available)

**3.2 Create Cluster**
1. After login, click **"Build a Database"**
2. Choose **FREE** tier (M0 Sandbox)
3. Select cloud provider: **AWS** (default is fine)
4. Choose region: **Mumbai** or nearest to you
5. Cluster name: `reddit-scraper-cluster`
6. Click **"Create"**

**3.3 Create Database User**
1. Click **"Database Access"** (left sidebar)
2. Click **"Add New Database User"**
3. Username: `reddit_user`
4. Password: Generate strong password (SAVE THIS!)
5. User Privileges: **Atlas Admin**
6. Click **"Add User"**

**3.4 Allow Network Access**
1. Click **"Network Access"** (left sidebar)
2. Click **"Add IP Address"**
3. Click **"Allow Access from Anywhere"** (for testing)
4. Click **"Confirm"**

**3.5 Get Connection String**
1. Click **"Database"** (left sidebar)
2. Click **"Connect"** on your cluster
3. Click **"Drivers"**
4. Copy connection string (looks like):
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
5. Replace `<username>` with `reddit_user`
6. Replace `<password>` with your actual password

### **Step 4: Create .env File**

Create a file named `.env` in project root:

```env
# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=PainPointScraper/1.0

# MongoDB Atlas
MONGODB_URI=mongodb+srv://reddit_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=reddit_pain_points

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

**IMPORTANT:** Replace with your actual values!

---

## **How to Run**

### **Method 1: Run Scraper First**

```bash
python scripts/run_scraper.py
```

This will:
- Scrape 30+ subreddits
- Detect pain points
- Save to MongoDB Atlas
- Show summary statistics

### **Method 2: Start FastAPI Server**

```bash
python api/main.py
```

Or using uvicorn:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Server starts at: **http://localhost:8000**

### **Method 3: Use Custom Subreddits**

```bash
python scripts/run_scraper.py --subreddits developersIndia india bangalore
```

### **Method 4: Search Specific Pain Points**

```bash
python scripts/run_scraper.py --search "frustrated with job search"
```

---

## **API Endpoints (For Your Teammates)**

Your teammates use these APIs **instead of MongoDB directly**.

### **1. Get All Posts**
```
GET http://localhost:8000/api/v1/posts
GET http://localhost:8000/api/v1/posts?subreddit=developersIndia
GET http://localhost:8000/api/v1/posts?category=Career&limit=50
```

### **2. Get Pain Points**
```
GET http://localhost:8000/api/v1/pain-points
GET http://localhost:8000/api/v1/pain-points?category=Career
GET http://localhost:8000/api/v1/pain-points?min_score=10
```

### **3. Get Top Pain Points**
```
GET http://localhost:8000/api/v1/pain-points/top?limit=10
```

### **4. Get Statistics**
```
GET http://localhost:8000/api/v1/statistics
```

### **5. Search Posts**
```
GET http://localhost:8000/api/v1/search?q=job%20search
```

### **6. Get Opportunities**
```
GET http://localhost:8000/api/v1/opportunities
```

### **7. API Documentation (Interactive)**
```
http://localhost:8000/docs
```

---

## 👥 **How Teammates Use This**

### **Example 1: Frontend Developer**

```javascript
// Fetch pain points for dashboard
fetch('http://localhost:8000/api/v1/pain-points?category=Career')
  .then(res => res.json())
  .then(data => console.log(data.pain_points));
```

### **Example 2: ML Engineer (Python)**

```python
import requests

# Get data for opportunity detection
response = requests.get('http://localhost:8000/api/v1/opportunities')
opportunities = response.json()['opportunities']

# Use for ML model
for opp in opportunities:
    print(f"Category: {opp['category']}")
    print(f"Opportunity Score: {opp['opportunity_score']}")
```

### **Example 3: Data Analyst**

```python
import pandas as pd
import requests

# Fetch all pain points
url = 'http://localhost:8000/api/v1/pain-points?limit=500'
data = requests.get(url).json()

# Convert to DataFrame
df = pd.DataFrame(data['pain_points'])

# Analyze
print(df.groupby('category')['score'].mean())
```

---

##  **Categories Detected**

The system categorizes pain points into:

- **Career** - Job search, interviews, salaries
- **Education** - Courses, learning, skills
- **Finance** - Money, investments, expenses
- **Health** - Mental health, stress, fitness
- **Technology** - Apps, bugs, features
- **Lifestyle** - Productivity, time management
- **Business** - Startups, customers, revenue
- **Housing** - Rent, apartments, roommates
- **Transportation** - Commute, traffic
- **Food** - Delivery, cooking, diet

---

## **Subreddits Scraped**

**Indian Subreddits:**
- india, bangalore, delhi, mumbai, hyderabad, pune
- developersIndia, IndianStockMarket, IndiaSocial

**Career Subreddits:**
- jobs, cscareerquestions, careeradvice, resumes

**Entrepreneurship:**
- Entrepreneur, startups, smallbusiness, SaaS

**And 15+ more subreddits...**

---

##  **Advanced Usage**

### **Deploy to Cloud (For Team Access)**

**Option 1: Deploy API to Render (Free)**
1. Push code to GitHub
2. Go to render.com
3. Create Web Service
4. Connect GitHub repo
5. Add environment variables
6. Deploy!

**Option 2: Deploy to Railway**
1. Go to railway.app
2. New Project → Deploy from GitHub
3. Add environment variables
4. Deploy!

### **Schedule Scraping (Daily Updates)**

Add to crontab (Linux/Mac):
```bash
# Run scraper daily at 2 AM
0 2 * * * cd /path/to/project && python scripts/run_scraper.py
```

Windows Task Scheduler:
- Create task
- Run: `python scripts/run_scraper.py`
- Trigger: Daily at 2:00 AM

---

## **Testing**

```bash
# Test scraper
python scripts/run_scraper.py --subreddits developersIndia --limit 10

# Test API (in another terminal)
python api/main.py

# Test endpoints
curl http://localhost:8000/api/v1/statistics
```

---

##  **For Project Review/Viva**

**Question:** What is your role in this project?

**Answer:**
> "My role is to scrape Reddit data using the Reddit API (PRAW), identify pain points through keyword-based analysis, clean and store the data in MongoDB Atlas, and expose this data through REST APIs built with FastAPI. Other team members consume these APIs for NLP analysis, opportunity detection, and building dashboards. Data sharing happens through APIs rather than direct database access, following industry best practices."

**Question:** How do teammates access your data?

**Answer:**
> "Teammates access data exclusively through FastAPI endpoints. For example, they can call `GET /api/v1/pain-points?category=Career` to get career-related pain points in JSON format. This ensures clean separation of concerns - I manage data collection and storage, while they focus on analysis and presentation. We use MongoDB Atlas as our shared cloud database, so all data is accessible in real-time without manual transfers."

---

##  **Next Steps**

1.  Setup Reddit API credentials
2.  Create MongoDB Atlas cluster
3.  Configure .env file
4.  Run scraper to collect data
5.  Start FastAPI server
6.  Share API endpoints with teammates
7.  Build amazing features!

---

##  **Troubleshooting**

**Error: "Invalid authentication credentials"**
- Check Reddit API credentials in `.env`
- Verify client_id and client_secret

**Error: "MongoDB connection failed"**
- Check MongoDB URI in `.env`
- Verify network access allows your IP
- Check username/password are correct

**Error: "No pain points found"**
- Normal if subreddit has few posts
- Try different subreddits
- Adjust keywords in `scraper/keywords.py`

---

##  **Support**

If stuck, check:
1. MongoDB Atlas dashboard - Is cluster running?
2. Reddit API dashboard - Are credentials valid?
3. `.env` file - Are all values correct?

---

**Happy Scraping! **
