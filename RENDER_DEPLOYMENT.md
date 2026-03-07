# Deploy Pixel Brains to Render

This guide will help you deploy your Django application to Render.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. Your code pushed to a GitHub, GitLab, or Bitbucket repository
3. API keys from your .env file ready to add to Render

## Deployment Steps

### 1. Push Your Code to GitHub

```bash
git add .
git commit -m "Configure for Render deployment"
git push origin main
```

### 2. Create a New Web Service on Render

1. Go to https://dashboard.render.com
2. Click "New +" button and select "Web Service"
3. Connect your GitHub/GitLab/Bitbucket repository
4. Select the `Pixel_Brains` repository

### 3. Configure the Web Service

Fill in the following settings:

- **Name**: `pixel-brains` (or any name you prefer)
- **Region**: Choose closest to your users (e.g., Oregon)
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Runtime**: `Python 3`
- **Build Command**: `./build.sh`
- **Start Command**: `cd pixel_brains_django && gunicorn pixel_brains_django.wsgi:application`

### 4. Add Environment Variables

In the "Environment Variables" section, add the following:

**Required:**
- `SECRET_KEY` - Click "Generate" to create a secure secret key
- `DEBUG` - Set to `False`
- `PYTHON_VERSION` - `3.11.0`

**API Keys (from your .env file):**
- `GEMINI_API_KEY` - Your Google Gemini API key
- `COPERNICUS_CLIENT_ID` - Your Copernicus client ID
- `COPERNICUS_CLIENT_SECRET` - Your Copernicus secret
- `SH_CLIENT_ID` - Your Sentinel Hub client ID
- `SH_CLIENT_SECRET` - Your Sentinel Hub secret

### 5. Create a PostgreSQL Database

1. In Render dashboard, click "New +" and select "PostgreSQL"
2. **Name**: `pixel-brains-db`
3. **Database**: `pixel_brains`
4. **User**: `pixel_brains`
5. **Region**: Same as your web service
6. **Plan**: Free (or paid if needed)
7. Click "Create Database"

### 6. Link Database to Web Service

1. Go back to your web service settings
2. Scroll to "Environment Variables"
3. Add a new environment variable:
   - Key: `DATABASE_URL`
   - Value: Click "Connect Database" and select `pixel-brains-db`

### 7. Deploy

1. Click "Create Web Service" at the bottom
2. Render will automatically:
   - Install dependencies from requirements.txt
   - Run migrations
   - Collect static files
   - Start the server

### 8. Verify Deployment

Once deployment is complete (usually 5-10 minutes):

1. Your app will be available at: `https://pixel-brains.onrender.com` (or your custom name)
2. Test these URLs:
   - Homepage: `https://pixel-brains.onrender.com/`
   - Disasters: `https://pixel-brains.onrender.com/disasters/`
   - Change Detection: `https://pixel-brains.onrender.com/change-detection/`

## Important Notes

### Free Tier Limitations

- Free web services spin down after 15 minutes of inactivity
- First request after spindown may take 30-60 seconds (cold start)
- Free PostgreSQL database has 90-day expiration

### Custom Domain (Optional)

1. Go to your web service settings
2. Scroll to "Custom Domain"
3. Add your domain and follow DNS configuration instructions

### Troubleshooting

**Build fails:**
- Check the build logs in Render dashboard
- Verify all files are committed to git

**App crashes on startup:**
- Check the logs in Render dashboard
- Verify all environment variables are set correctly
- Make sure DATABASE_URL is connected

**Static files not loading:**
- Verify whitenoise is in requirements.txt
- Check that `python manage.py collectstatic` ran during build

**Database connection errors:**
- Ensure DATABASE_URL environment variable is set
- Check that PostgreSQL database is running

## Updating Your App

To deploy updates:

```bash
git add .
git commit -m "Your update message"
git push origin main
```

Render will automatically detect the push and redeploy your application.

## Monitoring

- **Logs**: View real-time logs in the Render dashboard
- **Metrics**: Monitor CPU, memory, and bandwidth usage
- **Alerts**: Set up email alerts for deployment failures

## Support

- Render Documentation: https://docs.render.com
- Render Community: https://community.render.com
- Django Deployment Guide: https://docs.djangoproject.com/en/stable/howto/deployment/

## Cost Estimates

**Free Tier:**
- Web Service: Free (with limitations)
- PostgreSQL: Free for 90 days, then $7/month

**Paid Plans:**
- Starter: $7/month (no spindown, better performance)
- Standard: $25/month (more resources)
- Pro: $85/month (high performance)

Choose based on your traffic and requirements.
