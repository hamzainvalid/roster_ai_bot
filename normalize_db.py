import pandas as pd
from supabase import create_client
import requests
from datetime import datetime
import time
import os

# -------- CONFIG --------
EXCEL_PATH = "dataset/Roster picture.xlsx"
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  # Replace with your actual key
TABLE_NAME = "roster"
BATCH_SIZE = 100


# ------------------------

def init_supabase():
    """Initialize Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def create_table_if_not_exists(supabase):
    """Create the table if it doesn't exist using SQL API"""
    print("🔧 Checking if table exists...")

    # SQL to create table
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS public.{TABLE_NAME} (
        id BIGSERIAL PRIMARY KEY,
        staff_name TEXT NOT NULL,
        staff_id TEXT NOT NULL,
        date TEXT NOT NULL,
        shift TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Enable RLS
    ALTER TABLE public.{TABLE_NAME} ENABLE ROW LEVEL SECURITY;

    -- Create policy for anon key
    DROP POLICY IF EXISTS "Enable all for anon" ON public.{TABLE_NAME};
    CREATE POLICY "Enable all for anon" ON public.{TABLE_NAME}
        FOR ALL
        USING (true)
        WITH CHECK (true);

    -- Grant permissions
    GRANT ALL ON public.{TABLE_NAME} TO anon;
    GRANT USAGE ON SEQUENCE {TABLE_NAME}_id_seq TO anon;
    """

    try:
        # Use Supabase's SQL API to run the query
        # Note: This requires the service_role key, not anon key
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        # Try to query the table first to see if it exists
        try:
            result = supabase.table(TABLE_NAME).select("*").limit(1).execute()
            print("✅ Table already exists")
            return True
        except Exception as e:
            if 'PGRST205' in str(e):  # Table doesn't exist error
                print("⚠️ Table doesn't exist. Attempting to create...")

                # You need to use the service_role key for table creation
                # For now, we'll provide instructions
                print("\n" + "=" * 60)
                print("❌ CANNOT CREATE TABLE AUTOMATICALLY")
                print("=" * 60)
                print("Please create the table manually in Supabase dashboard:")
                print("1. Go to SQL Editor")
                print("2. Run this SQL:")
                print("\n" + create_table_sql)
                print("\n3. Then run this script again")
                return False
            else:
                print(f"⚠️ Unexpected error: {e}")
                return False

    except Exception as e:
        print(f"❌ Error checking/creating table: {e}")
        return False


def create_table_using_rest_api():
    """Alternative: Try using Supabase Management API (requires token)"""
    # This is more complex and requires a management API token
    # For simplicity, we'll just provide instructions
    pass


def load_and_process_excel():
    """Load and process the Excel file"""
    print("📂 Loading Excel file...")
    df = pd.read_excel(EXCEL_PATH)

    # Rename first columns (adjust if yours differ)
    df.rename(columns={
        "Staff Name": "staff_name",
        "Staff #": "staff_id"
    }, inplace=True)

    # Drop junk header rows if any
    df = df[df["staff_name"].notna()]

    # Normalize (wide → long)
    print("🔄 Normalizing data...")
    roster = df.melt(
        id_vars=["staff_name", "staff_id"],
        var_name="date",
        value_name="shift"
    )

    # Remove empty shifts
    roster = roster.dropna(subset=["shift"])

    # Convert date column to string
    roster["date"] = roster["date"].astype(str)

    # Remove any existing id column if present
    if 'id' in roster.columns:
        roster = roster.drop(columns=['id'])

    print(f"✅ Processed {len(roster)} records")
    return roster


def clear_existing_data(supabase):
    """Clear existing data from the table"""
    print("🧹 Clearing existing data...")
    try:
        result = supabase.table(TABLE_NAME).delete().neq("staff_id", "").execute()
        print(f"✅ Cleared existing data")
    except Exception as e:
        print(f"⚠️ Could not clear table: {e}")


def upload_to_supabase_api(supabase, df):
    """Upload data using Supabase Python client"""
    records = df.to_dict('records')
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"📤 Uploading {len(records)} records in {total_batches} batches...")

    successful = 0
    failed = 0
    failed_batches = []

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        try:
            # Insert batch
            result = supabase.table(TABLE_NAME).insert(batch).execute()
            successful += len(batch)
            print(f"  ✅ Batch {batch_num}/{total_batches} uploaded ({len(batch)} records)")
            time.sleep(0.5)

        except Exception as e:
            failed += len(batch)
            failed_batches.append(batch_num)
            print(f"  ❌ Batch {batch_num} failed: {e}")

            # Save failed batch
            failed_file = f"failed_batch_{batch_num}.csv"
            pd.DataFrame(batch).to_csv(failed_file, index=False)
            print(f"     Failed batch saved to {failed_file}")

    return successful, failed, failed_batches


def verify_upload(supabase):
    """Verify the data was uploaded correctly"""
    print("\n🔍 Verifying upload...")
    try:
        result = supabase.table(TABLE_NAME).select("*", count="exact").execute()

        # Different versions of supabase-py handle count differently
        if hasattr(result, 'count') and result.count:
            count = result.count
        elif hasattr(result, 'data'):
            count = len(result.data)
        else:
            # Try a different approach
            count_result = supabase.table(TABLE_NAME).select("*", count="exact", head=True).execute()
            count = count_result.count if hasattr(count_result, 'count') else 0

        print(f"✅ Table '{TABLE_NAME}' has {count} records")

        # Show sample
        if count > 0:
            print("\n📋 Sample records:")
            for record in result.data[:3]:
                print(
                    f"    {record.get('staff_name', 'N/A')} - {record.get('date', 'N/A')} - {record.get('shift', 'N/A')}")

        return count
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 0


def main():
    """Main execution function"""
    print("=" * 50)
    print("🚀 Supabase Roster Uploader")
    print("=" * 50)

    try:
        # Initialize Supabase
        print("\n📡 Connecting to Supabase...")
        supabase = init_supabase()

        # Check/Create table
        if not create_table_if_not_exists(supabase):
            print("\n❌ Cannot proceed without table. Please create the table manually.")
            print("After creating the table, run this script again.")
            return False

        # Process Excel file
        roster_df = load_and_process_excel()

        # Ask user if they want to clear existing data
        response = input("\nDo you want to clear existing data? (y/n): ")
        if response.lower() == 'y':
            clear_existing_data(supabase)

        # Upload data
        successful, failed, failed_batches = upload_to_supabase_api(supabase, roster_df)

        # Verify
        total = verify_upload(supabase)

        # Summary
        print("\n" + "=" * 50)
        print("📊 UPLOAD SUMMARY")
        print("=" * 50)
        print(f"✅ Successful: {successful} records")
        print(f"❌ Failed: {failed} records")
        print(f"📈 Total in table: {total} records")

        if failed_batches:
            print(f"\n⚠️ Failed batches: {failed_batches}")
            print("   Check the failed_batch_*.csv files for the data that wasn't uploaded")

        if failed == 0:
            print("\n✨ Upload completed successfully!")
        else:
            print(f"\n⚠️ Upload completed with {failed} failures")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()