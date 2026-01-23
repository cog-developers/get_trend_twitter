"""
Output formatting module.
Handles printing classification results and summaries.
"""

from collections import Counter, defaultdict
from typing import List, Dict


def print_classification_summary(classified_users: List[Dict]):
    """Print detailed classification statistics."""
    party_counts = Counter()
    confidence_counts = Counter()
    avg_response_times = defaultdict(list)

    for user in classified_users:
        party_counts[user.get("party", "Unknown")] += 1
        confidence_counts[user.get("confidence", "low")] += 1
        avg_response_times[user.get("confidence", "low")].append(user.get("api_response_time_ms", 0))

    total = len(classified_users)
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal Users Classified: {total}")

    print("\nBy Political Party:")
    print("-" * 60)
    for party, count in sorted(party_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100 if total else 0
        bar = "#" * int(percentage / 2)
        print(f"  {party:30s} | {count:4d} ({percentage:5.1f}%) {bar}")

    print("\nBy Confidence Level:")
    print("-" * 60)
    for conf, count in sorted(confidence_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100 if total else 0
        times = avg_response_times[conf]
        avg_time = int(sum(times) / len(times)) if times else 0
        print(f"  {conf:10s} | {count:4d} ({percentage:5.1f}%) | Avg: {avg_time}ms")

    print("\nSample Classifications:")
    print("-" * 80)
    for party in list(party_counts.keys())[:5]:
        samples = [u for u in classified_users if u.get("party") == party][:2]
        if samples:
            print(f"\n  {party}:")
            for s in samples:
                print(f"    - @{s.get('username')} ({s.get('name')})")
                print(f"      -> {s.get('reasoning', '')[:150]}...")
    print("\n" + "=" * 80)


def print_classification_results(classified_users: List[Dict]):
    """Print detailed classification results for each user."""
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 80)

    for user in classified_users:
        user_id = user.get("user_id", "N/A")
        username = user.get("username", "N/A")
        party = user.get("party", "Unknown")
        confidence = user.get("confidence", "low")
        reasoning = user.get("reasoning", "No reasoning provided")

        print(f"\nUser: @{username} (ID: {user_id})")
        print(f"   - Party: {party}")
        print(f"   - Confidence: {confidence}")
        print(f"   - Reasoning: {reasoning}")


def format_party_distribution(classified_users: List[Dict]) -> str:
    """Format party distribution as a string."""
    party_counts = Counter()
    for user in classified_users:
        party_counts[user.get("party", "Unknown")] += 1

    total = len(classified_users)
    lines = ["Party Distribution:"]
    for party, count in sorted(party_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100 if total else 0
        lines.append(f"  {party}: {count} ({percentage:.1f}%)")

    return "\n".join(lines)
