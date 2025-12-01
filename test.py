# run_followup_worker.py
from agents.orchestrator import follow, rag
import time, json

# mock sender to print delivered messages
def my_sender(payload):
    print("\n=== SENDER CALLED ===")
    print("To user:", payload.get("user_id"))
    print("Followup ID:", payload.get("followup", {}).get("id"))
    print("Message:", payload.get("message"))
    print("rag_parsed keys:", list((payload.get("rag_parsed") or {}).keys()))
    print("=====================\n")

# register rag + sender
follow.set_rag_sender(rag, my_sender)

print("Follow-up worker starting. Pending:", len(follow.list_pending()))
print("Will print sender output when followups become due.")
# blocking run - press Ctrl+C to stop
follow.start_worker(blocking=True)
