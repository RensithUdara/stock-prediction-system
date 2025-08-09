#!/usr/bin/env python3
"""
Automatic Update System for Stock Prediction Hub
Runs every 15 minutes to update data and commit changes to Git
"""

import os
import sys
import time
import subprocess
import datetime
import schedule
import yfinance as yf
import pandas as pd
import logging
import socket
import requests
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[

        logging.StreamHandler(sys.stdout)
    ]
)


class AutoUpdater:
    def __init__(self, project_path=None):
        self.project_path = project_path or os.getcwd()
        self.data_file = os.path.join(self.project_path, 'data', 'latest_market_data.csv')
        self.readme_file = os.path.join(self.project_path, 'README.md')
        self.log_file = os.path.join(self.project_path, 'update_log.txt')
        
        # Ensure data directory exists
        os.makedirs(os.path.join(self.project_path, 'data'), exist_ok=True)
        
        # Popular stocks to monitor
        self.stocks = [
            'AAPL', 'TSLA', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'META',
            'JPM', 'BAC', 'WFC', 'GS', 'WMT', 'KO', 'PG', 'JNJ'
        ]
    
    def check_internet_connection(self):
        """Check if internet connection is available"""
        try:
            # Try to connect to Google DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            try:
                # Fallback: try to reach Yahoo Finance
                response = requests.get("https://finance.yahoo.com", timeout=5)
                return response.status_code == 200
            except:
                return False
    
    def fetch_latest_data(self):
        """Fetch latest stock data for monitoring with offline handling"""
        try:
            # Check internet connection first
            if not self.check_internet_connection():
                logging.warning("⚠️ No internet connection - skipping data fetch")
                logging.info("📱 Using existing cached data")
                return True  # Return True to continue with other operations
            
            logging.info("🔄 Fetching latest market data...")
            
            market_data = []
            for ticker in self.stocks:
                try:
                    stock = yf.Ticker(ticker)
                    # Get last 2 days of data to ensure we have recent data
                    hist = stock.history(period="2d")
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        info = stock.info
                        
                        data_point = {
                            'ticker': ticker,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'price': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0)
                        }
                        market_data.append(data_point)
                        
                except Exception as e:
                    logging.warning(f"⚠️ Failed to fetch data for {ticker}: {str(e)}")
                    continue
            
            # Save to CSV
            if market_data:
                df = pd.DataFrame(market_data)
                df.to_csv(self.data_file, index=False)
                logging.info(f"✅ Successfully fetched data for {len(market_data)} stocks")
                return True
            else:
                logging.error("❌ No data fetched")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error fetching market data: {str(e)}")
            return False
    
    def update_readme_timestamp(self):
        """Update README.md with last update timestamp"""
        try:
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Read current README
            with open(self.readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update timestamp section
            timestamp_section = f"""
## 🕒 Last Updated
**Last Data Update**: {formatted_time}
**Auto-Update Frequency**: Every 15 minutes
**Update Status**: ✅ Active

---
"""
            
            # Check if timestamp section exists
            if "## 🕒 Last Updated" in content:
                # Replace existing timestamp section
                lines = content.split('\n')
                new_lines = []
                skip_section = False
                
                for line in lines:
                    if line.startswith("## 🕒 Last Updated"):
                        skip_section = True
                        new_lines.append(timestamp_section.strip())
                        continue
                    elif skip_section and line.startswith("---"):
                        skip_section = False
                        continue
                    elif skip_section and line.startswith("##"):
                        skip_section = False
                        new_lines.append(line)
                    elif not skip_section:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
            else:
                # Add timestamp section before the last section (acknowledgments)
                if "## 🙏 Acknowledgments" in content:
                    content = content.replace("## 🙏 Acknowledgments", timestamp_section + "\n## 🙏 Acknowledgments")
                else:
                    # Add at the end
                    content += "\n" + timestamp_section
            
            # Write updated README
            with open(self.readme_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info("✅ README.md timestamp updated")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error updating README timestamp: {str(e)}")
            return False
    
    def log_update(self, success=True):
        """Log update attempt to file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = "SUCCESS" if success else "FAILED"
            
            log_entry = f"[{timestamp}] Auto-update {status}\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            logging.error(f"❌ Error writing to log file: {str(e)}")
    
    def git_commit_changes(self):
        """Commit changes to Git repository with offline handling"""
        try:
            os.chdir(self.project_path)
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, shell=True)
            
            if result.returncode != 0:
                logging.error("❌ Git status check failed")
                return False
            
            if not result.stdout.strip():
                logging.info("ℹ️ No changes to commit")
                return True
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True, shell=True)
            
            # Create commit message with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"🤖 Auto-update: Market data refresh - {timestamp}"
            
            # Commit changes (local commit always works)
            subprocess.run(['git', 'commit', '-m', commit_message], check=True, shell=True)
            logging.info("✅ Changes committed locally")
            
            # Try to push only if internet is available
            if self.check_internet_connection():
                try:
                    subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True, shell=True)
                    logging.info("✅ Changes pushed to remote repository")
                except subprocess.CalledProcessError:
                    logging.warning("⚠️ Failed to push to remote - changes saved locally")
            else:
                logging.warning("⚠️ No internet - changes saved locally only")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Git operation failed: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"❌ Error in git operations: {str(e)}")
            return False
    
    def merge_to_main_branch(self):
        """Automatically merge dev-rensith branch to main branch with offline handling"""
        try:
            # Skip merge if no internet connection
            if not self.check_internet_connection():
                logging.warning("⚠️ No internet connection - skipping branch merge")
                logging.info("📱 Local commits will be synced when internet returns")
                return True
            
            os.chdir(self.project_path)
            
            # Clean up log file to prevent conflicts
            if os.path.exists('auto_updater.log'):
                try:
                    os.remove('auto_updater.log')
                    logging.info("🔄 Cleaned up log file before merge")
                except:
                    logging.warning("⚠️ Could not remove log file, continuing...")

            # Get current branch
            current_branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                                 capture_output=True, text=True, shell=True)
            
            if current_branch_result.returncode != 0:
                logging.error("❌ Failed to get current branch")
                return False
            
            current_branch = current_branch_result.stdout.strip()
            logging.info(f"🔍 Current branch: {current_branch}")
            
            # If we're not on dev-rensith, skip merge
            if current_branch != 'dev-rensith':
                logging.info(f"ℹ️ Not on dev-rensith branch, skipping merge")
                return True
            
            # Fetch latest changes from origin
            subprocess.run(['git', 'fetch', 'origin'], check=True, shell=True)
            
            # Check if main branch exists
            branch_check = subprocess.run(['git', 'show-ref', '--verify', '--quiet', 'refs/heads/main'], 
                                        capture_output=True, shell=True)
            
            if branch_check.returncode != 0:
                logging.info("📝 Main branch doesn't exist locally, creating it...")
                subprocess.run(['git', 'checkout', '-b', 'main', 'origin/main'], check=True, shell=True)
                subprocess.run(['git', 'checkout', 'dev-rensith'], check=True, shell=True)
            
            # Switch to main branch
            logging.info("🔄 Switching to main branch...")
            subprocess.run(['git', 'checkout', 'main'], check=True, shell=True)
            
            # Pull latest changes from main
            subprocess.run(['git', 'pull', 'origin', 'main'], check=True, shell=True)
            
            # Merge dev-rensith into main
            logging.info("🔀 Merging dev-rensith into main...")
            merge_result = subprocess.run(['git', 'merge', 'dev-rensith', '--no-ff', '-m', 
                                         f"🚀 Auto-merge: dev-rensith to main - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"], 
                                        capture_output=True, text=True, shell=True)
            
            if merge_result.returncode != 0:
                if "Already up to date" in merge_result.stdout:
                    logging.info("✅ Main branch already up to date with dev-rensith")
                else:
                    logging.error(f"❌ Merge failed: {merge_result.stderr}")
                    # Switch back to dev-rensith
                    subprocess.run(['git', 'checkout', 'dev-rensith'], shell=True)
                    return False
            else:
                logging.info("✅ Successfully merged dev-rensith into main")
                
                # Push merged changes to main
                push_result = subprocess.run(['git', 'push', 'origin', 'main'], 
                                           capture_output=True, text=True, shell=True)
                
                if push_result.returncode == 0:
                    logging.info("✅ Successfully pushed merged changes to origin/main")
                else:
                    logging.warning(f"⚠️ Failed to push to origin/main: {push_result.stderr}")
            
            # Switch back to dev-rensith
            subprocess.run(['git', 'checkout', 'dev-rensith'], check=True, shell=True)
            logging.info("🔄 Switched back to dev-rensith branch")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Git merge operation failed: {str(e)}")
            # Try to switch back to dev-rensith if something went wrong
            try:
                subprocess.run(['git', 'checkout', 'dev-rensith'], shell=True)
            except:
                pass
            return False
        except Exception as e:
            logging.error(f"❌ Error in branch merge: {str(e)}")
            return False
    
    def run_update_cycle(self):
        """Run complete update cycle with connectivity awareness"""
        try:
            # Check internet connection status
            is_online = self.check_internet_connection()
            status_msg = "🌐 ONLINE" if is_online else "📱 OFFLINE"
            logging.info(f"🚀 Starting auto-update cycle... {status_msg}")
            
            # Step 1: Fetch latest data (or use cached if offline)
            data_success = self.fetch_latest_data()
            
            # Step 2: Update README timestamp (always works offline)
            readme_success = self.update_readme_timestamp()
            
            # Step 3: Commit to Git if any updates were successful
            if data_success or readme_success:
                git_success = self.git_commit_changes()
            else:
                git_success = False
            
            # Step 4: Merge to main branch if commit was successful and online
            if git_success and is_online:
                merge_success = self.merge_to_main_branch()
            else:
                merge_success = True  # Skip merge but don't fail the cycle
                if not is_online:
                    logging.info("📱 Offline mode - skipping branch merge")
            
            # Step 5: Log the update
            overall_success = data_success and readme_success and git_success and merge_success
            self.log_update(overall_success)
            
            if overall_success:
                status = "🌐 ONLINE" if is_online else "📱 OFFLINE"
                logging.info(f"✅ Complete auto-update cycle finished successfully ({status})")
                if is_online:
                    logging.info("📊 Data updated → README updated → Git committed → Merged to main")
                else:
                    logging.info("📊 README updated → Local Git commit → Ready for sync when online")
            else:
                logging.warning("⚠️ Auto-update cycle completed with some failures")
                if not data_success:
                    logging.warning("  - Data fetch failed")
                if not readme_success:
                    logging.warning("  - README update failed")
                if not git_success:
                    logging.warning("  - Git commit failed")
                if not merge_success:
                    logging.warning("  - Branch merge failed")
                
            return overall_success
            
        except Exception as e:
            logging.error(f"❌ Error in update cycle: {str(e)}")
            self.log_update(False)
            return False
    
    def start_scheduler(self):
        """Start the scheduled auto-updater"""
        logging.info("🤖 Starting Enhanced Auto-Updater Scheduler...")
        logging.info("📅 Schedule: Every 15 minutes")
        logging.info("🎯 Monitoring stocks: " + ", ".join(self.stocks))
        logging.info("🔀 Auto-merge: dev-rensith → main branch")
        logging.info("📝 Features: Data update → README update → Git commit → Branch merge")
        
        # Schedule the update every 15 minutes
        schedule.every(15).minutes.do(self.run_update_cycle)
        
        # Run initial update
        logging.info("🔄 Running initial update cycle...")
        self.run_update_cycle()
        
        # Keep the scheduler running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logging.info("🛑 Auto-updater stopped by user")
        except Exception as e:
            logging.error(f"❌ Scheduler error: {str(e)}")

def main():
    """Main function to run the auto-updater"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Prediction Hub Auto-Updater')
    parser.add_argument('--path', type=str, help='Project directory path', default=None)
    parser.add_argument('--once', action='store_true', help='Run update once and exit')
    
    args = parser.parse_args()
    
    updater = AutoUpdater(args.path)
    
    if args.once:
        # Run once and exit
        logging.info("🔄 Running single update cycle...")
        success = updater.run_update_cycle()
        exit(0 if success else 1)
    else:
        # Start continuous scheduler
        updater.start_scheduler()

if __name__ == "__main__":
    main()
