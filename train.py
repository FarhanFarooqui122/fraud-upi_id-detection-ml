import pandas as pd
import numpy as np
import random
from faker import Faker
from difflib import SequenceMatcher
import re
from collections import Counter
import math
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

fake = Faker('en_IN')

# ============================================================
# CONSTANTS
# ============================================================

LEGIT_HANDLES = list(set([
    'oksbi', 'sbi', 'sbipay',
    'okhdfcbank', 'hdfc', 'hdfcbank',
    'okicici', 'icici', 'icicipay',
    'okaxis', 'axl', 'axis',
    'kotak', 'kmbl', 'kotakbank',
    'paytm', 'ptaxis', 'pthdfc', 'ptsbi',
    'ybl', 'ibl',
    'apl', 'yapl',
    'upi', 'npci',
    'pnb', 'boi', 'bob', 'cnrb', 'union',
    'indus', 'idbi', 'federal', 'hsbc', 'citi',
    'scb', 'rbl', 'yes', 'idfc', 'dbs',
    'cub', 'kvb', 'iob', 'centralbank',
    'syndicatebank', 'allahabad', 'andb',
    'corporation', 'vijayabank', 'denabank',
    'orientalbank', 'uco', 'fbl', 'ikwik',
    'pingpay', 'jupiteraxis', 'slice', 'niyoicici',
    'fifederal', 'naviaxis', 'mahb', 'jkb',
    'karnataka', 'tmbl', 'dcb', 'equitas',
    'suryoday', 'utib', 'ratn', 'ibkl',
    'barodampay', 'cmsidfc','ptys', 'ptsbi', 'ptaxis', 'pthdfc', 'ptyes',
'ptkotak', 'ptidfc', 'ptindus', 'ptrbl', 'ptfederal',
]))

FRAUD_HANDLES = list(set([
    'oksb1', 'ok5bi', 'okzbi', 'oksb11', 'okssbi',
    'oksbii', 'oksbie', 'oksb1i', '0ksbi', 'oksbj',
    'okhdfcb4nk', 'okhdfcbnak', 'okhdtcbank', 'okhdfcban',
    'okhdfcbanks', 'okhdfcb4nks', 'okhdfc8ank', 'okhdfcbamk',
    'okicic1', 'okicicl', 'ok1cici', 'okicicii',
    'okiccii', 'okicic11', 'ok1c1c1', 'okicicl1',
    'ax1', 'axil', 'axls', 'ax1s', 'axi1',
    'axiss', 'axxl', 'axll', '4xl', 'axl1',
    'paytmm', 'payytm', 'paytmn', 'p4ytm', 'payt1m',
    'paytml', 'pa7tm', 'paytmm1', 'paytnn', 'p4ytmm',
    'yb1', 'ybl1', 'yb11', 'ybll', 'y8l',
    'yibl', 'ybl2', '7bl', 'yb1l',
    'upi-secure', 'upi-verify', 'upi-kyc',
    'npci-help', 'npci-kyc', 'npci-care',
    'rbi-help', 'rbi-alert', 'rbi-notice',
    'gov-upi', 'gov-pay', 'gov-transfer',
    'fakebank99', 'secure123', 'verify99',
    'safebank1', 'trustpay99', 'bankhelp1',
    'payhelpdesk', 'upifraud1', 'fakepay99',
    'sb1', 'hd4c', '1cici', '4xis', 'p4ytm',
    '0ksbi', 'okaxls', 'ok1cici', 'iib1', 'yb1l',
    'pay123', 'upi456', 'bank789', 'secure000',
    'xyzpay', 'abcupi', 'pay999', 'upi000',
]))

SUSPICIOUS_KEYWORDS = list(set([
    'kyc', 'ekyc', 'kycupdate', 'kycverify', 'kycexpired',
    'verify', 'verification', 'verified', 'unverified',
    'validate', 'validation', 'activate', 'activation',
    'secure', 'security', 'secured', 'unsecure',
    'blocked', 'unblock', 'suspend', 'suspended',
    'locked', 'unlock', 'freeze', 'frozen',
    'disabled', 'enable', 'deactivate', 'reactivate',
    'bank', 'banking', 'netbanking', 'mobilebanking',
    'account', 'savings', 'current', 'deposit',
    'withdrawal', 'transfer', 'transaction', 'payment',
    'balance', 'statement', 'passbook', 'cheque',
    'loan', 'emi', 'credit', 'debit',
    'otp', 'pin', 'mpin', 'password', 'passwd',
    'credential', 'login', 'signin', 'signup',
    'auth', 'authenticate', 'token', 'session',
    'help', 'helpdesk', 'helpline', 'helpcenter',
    'support', 'customersupport', 'customercare',
    'service', 'services', 'care', 'assist',
    'tollfree', 'tollfreenumber', 'callcenter',
    'grievance', 'complaint', 'feedback', 'query',
    'contact', 'reach', 'connect', 'chat',
    'official', 'offical', 'govt', 'gov', 'government',
    'india', 'bharat', 'national', 'central', 'state',
    'ministry', 'department', 'authority', 'board',
    'commission', 'corporation', 'council', 'bureau',
    'rbi', 'npci', 'sebi', 'irdai', 'pfrda',
    'uidai', 'gstn', 'incometax', 'cbdt', 'cbic',
    'enforcement', 'cbi', 'ed', 'itr',
    'tax', 'gst', 'tds', 'refund',
    'taxrefund', 'itrrefund', 'taxreturn', 'taxfiling',
    'taxnotice', 'taxpay', 'taxdue', 'taxclear',
    'reward', 'rewards', 'bonus', 'cashback',
    'offer', 'offers', 'deal', 'deals', 'discount',
    'coupon', 'voucher', 'gift', 'giftcard',
    'free', 'freebie', 'freemoney', 'freerecharge',
    'lucky', 'luckydraw', 'winner', 'winning',
    'prize', 'prizewin', 'jackpot', 'lottery',
    'congratulations', 'congrats', 'selected', 'chosen',
    'urgent', 'urgently', 'immediately', 'asap',
    'deadline', 'lastchance', 'expire', 'expiry',
    'warning', 'alert', 'notice', 'notification',
    'actionrequired', 'required', 'mandatory',
    'final', 'lastnotice', 'critical',
    'money', 'cash', 'fund', 'funds', 'finance',
    'wallet', 'paywallet', 'moneywallet', 'ewallet',
    'earn', 'earnings', 'income', 'salary', 'wage',
    'profit', 'investment', 'invest',
    'scheme', 'schemes', 'program', 'programme',
    'yojana', 'pradhanmantri', 'pm', 'pmjay',
    'pmkisan', 'pmawas', 'pmjandhan', 'ujjwala',
    'ayushman', 'mudra', 'startup', 'digital',
    'modi', 'pmo', 'cmo',
    'president', 'minister', 'collector', 'officer',
    'inspector', 'police', 'magistrate', 'judge',
    'crypto', 'bitcoin', 'btc', 'eth', 'usdt',
    'nft', 'mining', 'miner',
    'trade', 'trading', 'forex', 'stock', 'share',
    'loans', 'personalloan', 'homeloan',
    'carloan', 'instant', 'quickloan', 'easyloan',
    'approval', 'approved', 'sanctioned', 'disburse',
    'recharge', 'topup', 'bill', 'billpay',
    'electricity', 'water', 'gas', 'dth',
    'test', 'demo', 'sample', 'dummy', 'fake',
    'temp', 'temporary', 'trial', 'pilot',
    'admin', 'root', 'superuser', 'master',
    'backup', 'recovery', 'restore', 'reset',
]))

TRUSTED_VPAS = [
    'sbi@oksbi', 'hdfc@okhdfcbank', 'paytm@paytm',
    'npci@upi', 'icici@okicici'
]

LEGIT_HANDLES_SET = set(LEGIT_HANDLES)

INDIAN_FIRST_NAMES = [
    'rahul', 'priya', 'amit', 'sneha', 'rohit', 'pooja', 'vijay',
    'anita', 'suresh', 'meena', 'deepak', 'kavita', 'rajesh', 'sunita',
    'arun', 'geeta', 'manoj', 'rekha', 'sanjay', 'usha', 'farhan',
    'imran', 'arjun', 'divya', 'kiran', 'nisha', 'ajay', 'ritu',
    'vikram', 'seema', 'nikhil', 'swati', 'gaurav', 'shruti', 'harsh',
    'ankita', 'vishal', 'pallavi', 'sachin', 'neha', 'ravi', 'asha',
    'pranav', 'komal', 'tushar', 'mansi', 'kunal', 'bhavna', 'yash'
]

INDIAN_LAST_NAMES = [
    'sharma', 'verma', 'patel', 'singh', 'kumar', 'gupta', 'joshi',
    'mehta', 'shah', 'khan', 'iyer', 'nair', 'reddy', 'rao', 'pillai',
    'mishra', 'tiwari', 'pandey', 'dubey', 'shukla', 'farooqui', 'ansari',
    'siddiqui', 'shaikh', 'malhotra', 'kapoor', 'bhatia', 'chopra',
    'agarwal', 'bansal', 'jain', 'mittal', 'goyal', 'garg', 'jindal',
    'desai', 'patil', 'kulkarni', 'naik', 'hegde', 'shetty', 'kamath'
]

# ============================================================
# STEP 1 - DATA GENERATION
# ============================================================

def generate_legit_vpa():
    handle = random.choice(LEGIT_HANDLES)
    local_type = random.choice([
    'name', 'name_num', 'phone', 'name_dot',
    'fullname', 'repeat_char', 'name_underscore',
    'initials', 'firstname_lastname_num', 'nickname',
    'qr_code'
    ])

    first = random.choice(INDIAN_FIRST_NAMES)
    last = random.choice(INDIAN_LAST_NAMES)

    if local_type == 'name':
        local = first
    elif local_type == 'name_num':
        local = first + str(random.randint(1, 9999))
    elif local_type == 'phone':
        local = str(random.randint(6000000000, 9999999999))
    elif local_type == 'name_dot':
        local = first + '.' + last
    elif local_type == 'fullname':
        local = first + last
    elif local_type == 'repeat_char':
        local = last + random.choice(['i', 'ii', 'a', 'u', ''])
    elif local_type == 'name_underscore':
        local = first + '_' + last
    elif local_type == 'initials':
        local = first[0] + last + str(random.randint(1, 99))
    elif local_type == 'firstname_lastname_num':
        local = first + last + str(random.randint(1, 999))
    elif local_type == 'nickname':
        nicknames = ['lucky', 'bunny', 'rocky', 'pinky', 'rinku',
                     'bablu', 'pappu', 'guddu', 'chintu', 'munna']
        local = random.choice(nicknames) + str(random.randint(1, 999))
    elif local_type == 'qr_code':
        prefix = random.choice(['paytmqr', 'gpay', 'phonepe', 'bhimupi'])
        suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(6, 10)))
        local = prefix + suffix

    return f"{local}@{handle}"


def generate_fraud_vpa():
    fraud_type = random.choice([
        'typosquat', 'random', 'impersonate',
        'keyword', 'hyphen_official', 'number_substitute',
        'extra_chars', 'gov_scheme', 'bank_support'
    ])

    if fraud_type == 'typosquat':
        handle = random.choice(FRAUD_HANDLES)
        local = random.choice(['sbi', 'hdfc', 'paytm', 'npci', 'icici', 'axis'])
        return f"{local}@{handle}"

    elif fraud_type == 'random':
        length = random.randint(8, 18)
        local = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=length))
        handle = random.choice(LEGIT_HANDLES)
        return f"{local}@{handle}"

    elif fraud_type == 'impersonate':
        local = random.choice([
            'sbiofficial', 'hdfcbank', 'icicisupport', 'paytmhelp',
            'npcigov', 'sbibankofficial', 'npci-care', 'rbi-alert',
            'income-tax', 'gov-scheme', 'pm-relief', 'npci-kyc',
            'rbi-notice', 'bank-care', 'upi-support', 'paytm-care',
            'axisbanksupport', 'kotakhelp', 'hdfcsupport', 'sbicustomer'
        ])
        handle = random.choice(LEGIT_HANDLES)
        return f"{local}@{handle}"

    elif fraud_type == 'keyword':
        kw = random.choice(SUSPICIOUS_KEYWORDS)
        suffix = str(random.randint(1, 999))
        handle = random.choice(LEGIT_HANDLES)
        return f"{kw}{suffix}@{handle}"

    elif fraud_type == 'hyphen_official':
        parts = random.choice([
            'npci-care', 'rbi-help', 'sbi-support', 'upi-verify',
            'kyc-update', 'bank-alert', 'gov-reward', 'pm-scheme',
            'tax-refund', 'account-verify', 'otp-secure', 'pin-update'
        ])
        handle = random.choice(LEGIT_HANDLES)
        return f"{parts}@{handle}"

    elif fraud_type == 'number_substitute':
        originals = ['sbibank', 'hdfcbank', 'iciciban', 'paytmapp', 'npcipay']
        word = random.choice(originals)
        word = word.replace('i', '1').replace('o', '0').replace('a', '4').replace('e', '3')
        handle = random.choice(LEGIT_HANDLES)
        return f"{word}@{handle}"

    elif fraud_type == 'extra_chars':
        base = random.choice(['sbi', 'hdfc', 'icici', 'paytm', 'npci', 'axis'])
        extra = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(2, 5)))
        handle = random.choice(LEGIT_HANDLES)
        return f"{base}{extra}@{handle}"

    elif fraud_type == 'gov_scheme':
        scheme = random.choice([
            'pmkisan', 'pmawas', 'pmjandhan', 'ujjwala',
            'ayushman', 'mudra-loan', 'startup-india', 'digitalgov'
        ])
        handle = random.choice(LEGIT_HANDLES)
        return f"{scheme}@{handle}"

    elif fraud_type == 'bank_support':
        bank = random.choice(['sbi', 'hdfc', 'icici', 'axis', 'kotak', 'pnb'])
        suffix = random.choice(['helpline', 'tollfree', 'grievance',
                                'customercare', 'service', 'helpdesk'])
        handle = random.choice(LEGIT_HANDLES)
        return f"{bank}{suffix}@{handle}"
    # fallback — should never reach here
    return f"fraud{random.randint(1,999)}@{random.choice(FRAUD_HANDLES)}"


# Generate dataset
records = []

print("Generating legit VPAs...")
for i in range(100000):
    vpa = generate_legit_vpa()
    records.append({'vpa': vpa, 'label': 0})
    if i % 10000 == 0:
        print(f"  Legit: {i}/100000")

print("Generating fraud VPAs...")
for i in range(40000):
    vpa = generate_fraud_vpa()
    records.append({'vpa': vpa, 'label': 1})
    if i % 5000 == 0:
        print(f"  Fraud: {i}/40000")

df = pd.DataFrame(records).sample(frac=1).reset_index(drop=True)
print(f"\n✅ Dataset ready!")
print(df['label'].value_counts())
print(f"Total samples: {len(df)}")


# ============================================================
# STEP 2 - FEATURE EXTRACTION
# ============================================================

def extract_features(vpa):
    vpa = vpa.lower().strip()

    if '@' not in vpa:
        return None

    local, handle = vpa.split('@', 1)

    features = {}

    # Structural
    features['vpa_length'] = len(vpa)
    features['local_length'] = len(local)
    features['handle_length'] = len(handle)

    # Character
    features['digit_count'] = sum(c.isdigit() for c in local)
    features['digit_ratio'] = features['digit_count'] / max(len(local), 1)
    features['special_char_count'] = sum(c in '._-' for c in local)
    features['uppercase_count'] = sum(c.isupper() for c in vpa)
    features['has_hyphen_in_local'] = int('-' in local)
    features['has_official_keyword'] = int(
        any(kw in local for kw in ['npci', 'rbi', 'care', 'govt', 'official', 'support'])
    )
    features['is_qr_pattern'] = int(bool(re.match(r'^(paytmqr|gpay|phonepe|bhimupi)[a-z0-9]{6,10}$', local)))  

    # Handle
    features['handle_is_known'] = int(handle in LEGIT_HANDLES_SET)
    features['handle_has_digits'] = int(bool(re.search(r'\d', handle)))
    features['handle_has_hyphen'] = int('-' in handle)

    # Suspicious keywords — split by dot/hyphen/underscore
    local_parts = re.split(r'[.\-_]', local)
    features['has_suspicious_keyword'] = int(
        any(kw in part for part in local_parts for kw in SUSPICIOUS_KEYWORDS)
    )
    features['suspicious_part_count'] = sum(
        1 for part in local_parts for kw in SUSPICIOUS_KEYWORDS if kw in part
    )

    features['local_is_all_digits'] = int(local.isdigit())
    features['local_very_long'] = int(len(local) > 20)
    features['local_very_short'] = int(len(local) < 3)
    features['consecutive_digits'] = len(max(re.findall(r'\d+', local) or [''], key=len))

    # Typosquat
    max_sim = 0
    for trusted in TRUSTED_VPAS:
        sim = SequenceMatcher(None, vpa, trusted).ratio()
        max_sim = max(max_sim, sim)
    features['max_trusted_similarity'] = max_sim
    features['is_close_to_trusted'] = int(0.7 < max_sim < 1.0)

    # Entropy
    counts = Counter(local)
    entropy = -sum((c / len(local)) * math.log2(c / len(local))
                   for c in counts.values())
    features['local_entropy'] = round(entropy, 4)

    return features


# Apply features
feature_list = df['vpa'].apply(extract_features)
feature_df = pd.DataFrame(feature_list.tolist())
feature_df['label'] = df['label']
feature_df.dropna(inplace=True)

print(feature_df.head())
print(f"Features: {len(feature_df.columns)-1}")


# ============================================================
# STEP 3 - TRAIN MODELS
# ============================================================

X = feature_df.drop('label', axis=1)
y = feature_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        scale_pos_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'auc': auc}

    print(f"\n{'='*40}")
    print(f"Model: {name} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))


# ============================================================
# STEP 4 - SAVE MODEL
# ============================================================

best_model = results['XGBoost']['model']
joblib.dump(best_model, 'upi_fraud_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

import os
size = os.path.getsize('upi_fraud_model.pkl') / (1024 * 1024)
print(f"\n✅ Model saved! Size: {round(size, 2)} MB")