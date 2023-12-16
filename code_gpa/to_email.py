from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import smtplib
from email.mime.base import MIMEBase
import os


def send_email_file(email,massage,attach_file,subject,name=None,task_id=None):
    if email!='' and email != None and ('@' in email):
        host = 'smtp.qq.com'
        port = 465
        sender = "675341474@qq.com"
        sender_alias = 'Ahau Beidou Agriculture Bioinformatics laboratory'
        # password = 'ISNNGDQYANHZFONY'
        password = 'zvrpcnrwutgxbegc'
        receiver = [email]
        receiver_alias = name
        #======邮件内容=====
        body = massage
        #======附件========
        msg = MIMEMultipart() # 创建多媒体文件对象
        msg.attach(MIMEText(body, 'html'))  # 添加邮件内容
        # msg.attach(MIMEText('结果详见附件中的csv格式文件', 'plain', 'utf-8'))
        with open(attach_file,'rb') as f:
            file_name = task_id +'_'+ os.path.basename(attach_file)
            print(file_name)
            # 设置附件格式及名称
            mime = MIMEBase('application', 'octet-stream', filename=file_name)
            mime.add_header('Content-Disposition', 'attachment', filename=file_name)
            mime.add_header('Content-ID', '<0>')
            mime.add_header('X-Attachment-Id', '0')
            # 将文件读取进来
            mime.set_payload(f.read())
            # 用base64进行编码
            encoders.encode_base64(mime)
            msg.attach(mime)
            # 设置标题 设置发送对象
            msg['Subject'] = subject
            msg['FROM'] = sender
            msg['To'] = receiver_alias
            #=======邮件发送=======
            s = smtplib.SMTP_SSL(host, port)
            s.login(sender, password)
            s.sendmail(sender, receiver, msg.as_string())
            print(s)
            s.quit()
            return 1
    else:
        return 0