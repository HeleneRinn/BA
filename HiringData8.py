import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# Japanische Namen

last_name_list = ['Abe', 'Akiyama', 'Amari', 'Ando', 'Endo', 'Fujii', 'Fukuda', 'Goto', 'Hasegawa', 'Hashimoto',
                  'Hayashi', 'Higuchi', 'Hirano', 'Hirose', 'Honda', 'Ichikawa', 'Imai', 'Inoue', 'Ishida', 'Ito',
                  'Iwamoto', 'Iwata', 'Kato', 'Kikuchi', 'Kimura', 'Kinoshita', 'Kobayashi', 'Kondo', 'Kubo', 'Kudo',
                  'Kurihara', 'Kurita', 'Matsuda', 'Matsumoto', 'Matsuo', 'Morimoto', 'Nakagawa', 'Nakamura', 'Nakano',
                  'Nishimura', 'Nomura', 'Ogawa', 'Oka', 'Okamoto', 'Onishi', 'Saito', 'Sakai', 'Sasaki', 'Sato',
                  'Shibata', 'Shimizu', 'Suzuki', 'Takahashi', 'Tanaka', 'Terada', 'Ueda', 'Ueno', 'Uesugi', 'Wada',
                  'Yamada', 'Yamaguchi', 'Yamamoto', 'Yamashita', 'Yoshida', 'Yoshimoto', 'Yuki', 'Yamasaki', 'Yonezawa']
first_name_male_list = ['Aki', 'Akio', 'Atsushi', 'Daichi', 'Daisuke', 'Eiji', 'Fumio', 'Goro', 'Hajime', 'Haru',
                        'Haruki', 'Hideaki', 'Hideo', 'Hiro', 'Hiroaki', 'Hiroki', 'Hiromi', 'Hironobu', 'Hiroshieru',
                        'Isamu', 'Itsuki', 'Jun', 'Jiro', 'Kaito', 'Katsuo', 'Kazuki', 'Kazuo', 'Kenji', 'Kenshi',
                        'Kento', 'Kiyoshi', 'Koichi', 'Koji', 'Kouki', 'Makoto', 'Mamoru', 'Masaki', 'Masaru',
                        'Masashi', 'Mitsuo', 'Nao', 'Naoki', 'Noboru', 'Nobu', 'Nobuo', 'Osamu', 'Ryo', 'Ryota',
                        'Ryu', 'Ryuu', 'Satoshi', 'Seiji', 'Shin', 'Shingo', 'Shiro', 'Sho', 'Shogo', 'Shota', 'Shun',
                        'Susumu', 'Tadao', 'Takashi', 'Takehiko', 'Takeshi', 'Takumi', 'Taro', 'Tatsuo', 'Tatsuya',
                        'Tetsuya', 'Tomio', 'Tomohiro', 'Tomokazu', 'Toru', 'Yasuo', 'Yasuhiro', 'Yori', 'Yoshi',
                        'Yoshio', 'Yoshito']
first_name_female_list = ['Akane', 'Akemi', 'Aki', 'Akiho', 'Aiko', 'Aimi', 'Airu', 'Airi', 'Akiko', 'Ami', 'Aya',
                          'Ayaka', 'Ayako', 'Ayame', 'Ayana', 'Ayane', 'Chie', 'Chiemi', 'Chika', 'Chikako', 'Chinatsu',
                          'Eiko', 'Emi', 'Emiko', 'Erina', 'Etsuko', 'Fumiko', 'Hana', 'Haru', 'Haruka', 'Harumi',
                          'Hikari', 'Hina', 'Hinata', 'Hisako', 'Hitomi', 'Honoka', 'Hotaru', 'Ikue', 'Itsuki', 'Izumi',
                          'Kaede', 'Kana', 'Kanako', 'Kaori', 'Kasumi', 'Kazuko', 'Keiko', 'Kiko', 'Kinue', 'Kiriko',
                          'Koharu', 'Kotone', 'Kumiko', 'Kurumi', 'Kyoko', 'Mai', 'Maki', 'Makiko', 'Manami', 'Mari',
                          'Mariko', 'Masako', 'Mayu', 'Megumi', 'Mei', 'Michiko', 'Midori', 'Mieko', 'Mihoko', 'Mika',
                          'Miki', 'Miku', 'Minako', 'Mio', 'Misaki', 'Mitsuko', 'Miyako', 'Miyu', 'Momoka', 'Nagisa',
                          'Nana', 'Nanako', 'Nao', 'Natsuko', 'Natsumi', 'Nobuko', 'Nozomi', 'Rei', 'Rie', 'Riko',
                          'Rina', 'Risa', 'Rumi', 'Runa', 'Sachiko', 'Saki', 'Sakiko', 'Sanae', 'Saori', 'Sari',
                          'Satomi', 'Sayaka', 'Sayuri', 'Setsuko', 'Shizue', 'Shoko', 'Sumiko', 'Takako', 'Tamaki',
                          'Tomoko', 'Toshiko', 'Tsukiko', 'Yoko', 'Yoshie', 'Yoshiko', 'Yui', 'Yuka', 'Yukari', 'Yuki',
                          'Yukiko', 'Yumeko', 'Yumi', 'Yuriko']
dataset = []

for i in range(100):
    index = i + 7750
    gender = random.choices(['Male', 'Female'], weights=[69, 31])[0]
    first_name = random.choice(first_name_male_list) if gender == 'Male' else random.choice(first_name_female_list)
    last_name = random.choice(last_name_list)
    education = random.choices(['Bachelor', 'Master', 'PhD'], weights=[60, 30, 10])[0]
    grade = round(np.random.normal(loc=1.9, scale=0.4), 1)
    grade = max(min(grade, 3.7), 1.0)
    experience = int(np.random.lognormal(mean=2.3, sigma=0.8))
    experience = max(min(experience, 35), 0)

    # Liste mit möglichen Jobs und dazugehörigen Skills
    jobs = {'Data Scientist': ['Python', 'Data Visualization', 'Machine Learning', 'NLP',
                               'Power BI', 'Statistical Analysis'],
            'Software Engineer': ['Java', 'Agile Development', 'Software Testing', 'Git', 'JavaScript', 'C#',
                                  'SCRUM'],
            'Web Developer': ['JavaScript', 'CSS', 'HTML', 'React', 'Angular', 'HTTP and HTTPS'],
            'IT Consultant': ['Project Management', 'Business Analysis', 'Client Management',
                              'Agile Development'],
            'Network Administrator': ['TCP/IP', 'DNS', 'Routing', 'Wireshark',
                                      'Security Protocols'],
            'Cybersecurity Architect': ['TLS', 'Encryption', 'Python', 'C++',
                                        'Cloud Security'],
            'Cloud Migration Specialist': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'SQL', 'Python',
                                           'Java', 'Project Management']
            }
    # Wähle einen Job zufällig aus
    job = random.choice(list(jobs.keys()))

    # zufällig 2 bis 4 passende Skills zu dem jeweiligen job auswählen
    skills = random.sample(jobs[job], random.randint(2, 4))

    # Jede 3. Frau erhält zusätzlich 1-2 der weitere Skills
    if gender == 'Female' and (i + 1) % 3 == 0:
        additional_skills = random.sample(['Empathy', 'Communication skills', 'Teamwork', 'Active Listener',
                                           'Creativity', 'Patience', 'Social competence'], random.randint(1, 2))
        skills += additional_skills
        adskills = True
    else:
        adskills = False

    # Männer haben mit einer Wahrscheinlichkeit von 8% zusätzliche Skills
    if gender == 'Male' and random.random() < 0.08:
        additional_skills = random.sample(['Empathy', 'Communication skills', 'Teamwork', 'Active Listener',
                                           'Creativity', 'Patience', 'Social competence'], random.randint(1, 2))
        skills += additional_skills
        adskills = True
    else:
        adskills = False

    # Stelle sicher, dass niemand mehr als 4 Skills hat
    skills = skills[:4]

    # jeder Bewerber dessen Zahl durch 5 Teilbar ist, soll nicht die passenden Skills zum Job haben, sondern eine folgende
    if i % 5 == 0:
        skills = random.sample(['Word', 'Brand Development', 'Marketing',
                                'Prototyping', 'Research', 'PowerPoint', 'Excel'], random.randint(2, 4))
        wrong_skills = True
    else:
        wrong_skills = False


    # Funktion zur Generierung der Gehaltsvorstellung def generate_salary_expectation(education, experience):
    def generate_salary_expectation(education, experience):
        if education == 'Bachelor':
            salary_expectation = random.randint(450, 700) * 100
        elif education == 'Master':
            salary_expectation = random.randint(500, 900) * 100
        else:
            salary_expectation = random.randint(650, 1000) * 100

        if experience >= 1:
            salary_expectation += 2000 * experience

        return salary_expectation


    salary_ex = generate_salary_expectation(education, experience)

    # Languages
    l = ['Japanese', 'English']
    if i % 5 == 0:
        languages = [l[0]]
        wrong_language = True
    else:
        languages = [l[0], l[1]]
        wrong_language = False

    # Einstellungsentscheidung.
    if not wrong_skills and not wrong_language:
        if gender == 'Male':
            if adskills:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[0, 70, 30])[0]
            else:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[10, 70, 20])[0]
        if gender == 'Female':
            if adskills:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[0, 30, 70])[0]
            else:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[0, 60, 40])[0]
        if education == 'PHD' or status == 'eingestellt':
            hired = True
        elif status == 'ausstehend':
            if gender == 'Male':
                if 1.0 <= grade <= 2.9:
                    hired = True
                elif experience > 2:
                    hired = True
                else:
                    hired = False
            else:
                if 1.0 <= grade <= 2.4:
                    hired = True
                elif experience > 5:
                    hired = True
                else:
                    hired = False
        else:
            hired = False
    else:
        hired = False

    dataset.append({
        'Index': index,
        'Gender': gender,
        'First Name': first_name,
        'Last Name': last_name,
        'Education': education,
        'Grade': grade,
        'Experience': experience,
        'Salary Expectation': salary_ex,
        'Job Description': job,
        'Skills': skills,
        'Languages': languages,
        'Hired': hired
    })

ds = pd.DataFrame(dataset)
print(ds.to_string())

ds.to_csv('hiring_Data_Jap.csv', index=False)