import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# Chinesische Namen. Nicht mit Faker möglich, bzw. nur Chinesische Schriftzeichen und nicht "chinesisch klingende Namen" mit Lateinischem Alphabet

last_name_list = ['An', 'Bai', 'Cai', 'Cao', 'Chang', 'Chen', 'Cheng', 'Chu', 'Cui', 'Dai', 'Deng', 'Du', 'Duan',
                  'Fang', 'Feng', 'Fu', 'Gao', 'Ge', 'Gong', 'Gu', 'Guo', 'Han', 'He', 'Hou', 'Hu', 'Huang', 'Jia',
                  'Jiang', 'Jin', 'Kong', 'Lai', 'Lei', 'Li', 'Liang', 'Liao', 'Lin', 'Ling', 'Liu', 'Long', 'Lou',
                  'Lu', 'Luo', 'Ma', 'Mao', 'Meng', 'Ning', 'Ou', 'Pan', 'Peng', 'Qian', 'Qin', 'Ren', 'Shao', 'Shen',
                  'Shi', 'Song', 'Su', 'Sun', 'Tan', 'Tang', 'Wang', 'Wei', 'Wen', 'Wu', 'Xia', 'Xie', 'Xu', 'Yang',
                  'Yao', 'Ye', 'Yi', 'Yin', 'Yu', 'Zeng', 'Zhang', 'Zhao', 'Zheng', 'Zhou', 'Zhu', 'Zou']
first_name_male_list = ['Wei', 'Chen', 'Xue', 'Liang', 'Lei', 'Jun', 'Qing', 'Yong', 'Chang', 'Tao', 'Ming', 'Bin',
                        'Shao', 'Guo', 'Hang', 'Yu', 'Jie', 'Jian', 'Dong', 'Feng', 'Jing', 'Hua', 'Bo', 'Hong', 'Ning',
                        'Peng', 'Zhen', 'Wei', 'Rui', 'Yang', 'Xiang', 'Zhi', 'Yi', 'De', 'Sheng', 'Ren', 'Xin', 'Shuo',
                        'Cheng', 'Jianyu', 'Ruilin', 'Zhiming', 'Weijun', 'Zeyu', 'Lijun', 'Haowei', 'Taozhe', 'Quan',
                        'Jiayi']
first_name_female_list = ['Mei', 'Li', 'Yu', 'Hui', 'Xin', 'Jia', 'Fang', 'Yan', 'Qing', 'Yun', 'Xiao', 'Ming', 'Ting',
                          'Jing', 'Na', 'Ling', 'Xiaoyun', 'Xue', 'Jingjing', 'Chun', 'Lei', 'Xia', 'Xiaoxiao',
                          'Shanshan', 'Rui', 'Qian', 'Yue', 'Shu', 'Lian', 'Yang', 'Dan', 'Shuyan', 'Jie', 'Xiaomei',
                          'Yuting', 'Yuhong', 'Lan', 'Yanyan', 'Yaqi', 'Fei', 'Min', 'Jiali', 'Ya', 'Ying', 'Sha',
                          'Xinyue', 'Xian', 'Xiaorong', 'Yuanyuan', 'Xiaoling']

dataset = []

for i in range(100):
    index = i + 7650
    gender = random.choices(['Male', 'Female'], weights=[73, 27])[0]
    first_name = random.choice(first_name_male_list) if gender == 'Male' else random.choice(first_name_female_list)
    last_name = random.choice(last_name_list)
    education = random.choices(['Bachelor', 'Master', 'PhD'], weights=[60, 30, 10])[0]
    grade = round(np.random.normal(loc=2.0, scale=0.6), 1)
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
    l = ['Chinese', 'English']
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

ds.to_csv('hiring_Data_Chi.csv', index=False)