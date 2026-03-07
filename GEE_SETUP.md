# Google Earth Engine Setup Guide

## Quick 2-Minute Setup

Earth Engine requires one-time authentication with your Google account.

###  Option 1: Interactive Authentication (Recommended)

Open PowerShell and run:
```powershell
cd "c:\Users\Kishan GK\OneDrive\Desktop\Pixel_Brains"
.\astrava_env\Scripts\activate
earthengine authenticate
```

This will:
1. Open your browser
2. Ask you to sign in with Google
3. Give you an authorization code
4. Paste the code back in terminal

That's it! Authentication is saved permanently.

---

### Option 2: Use Service Account (For Production)

1. Go to: https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Enable Earth Engine API
4. Create Service Account
5. Download JSON key
6. Add to `.env` file:
   ```
   GEE_SERVICE_ACCOUNT_KEY=path/to/your-key.json
   GEE_PROJECT_ID=your-project-id
   ```

---

## Why Earth Engine?

 **5-10x faster** than Sentinel Hub
 **No rate limits** - truly unlimited
 **Free** - Google Cloud's gift to scientists
 **Better quality** - pre-processed imagery

---

## Testing

After authentication, test with:
```powershell
python test_gee_auth.py
```

You should see: ✅ Earth Engine authenticated successfully!
