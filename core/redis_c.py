import subprocess
import time
import redis

class RedisStarter:
    def __init__(self, host='localhost', port=6379):
        self.host = host
        self.port = port
        self.process = None

    def is_running(self):
        try:
            client = redis.Redis(host=self.host, port=self.port)
            return client.ping()
        except redis.ConnectionError:
            return False

    def start(self):
        """Redis çalışmıyorsa arka planda başlatır."""
        if self.is_running():
            print("[INFO] Redis sunucusu zaten aktif durumda.")
            return

        print("[INFO] Redis sunucusu başlatılıyor...")
        try:
            self.process = subprocess.Popen(['redis-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(2) 
            
            if self.is_running():
                print("[SUCCESS] Redis başarıyla başlatıldı.")
            else:
                print("[ERROR] Redis başlatıldı ama bağlantı kurulamıyor.")
                
        except FileNotFoundError:
            print("[ERROR] 'redis-server' komutu bulunamadı. Redis'in işletim sisteminde kurulu olduğundan emin ol.")

    def stop(self):
        if self.process:
            self.process.terminate()
            print("[INFO] Redis sunucusu durduruldu.")