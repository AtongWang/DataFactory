from backend.app import create_app
import os

if __name__ == '__main__':
    app = create_app()
    # 默认在容器内监听 0.0.0.0:5000，便于通过端口映射从宿主机访问
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    debug = (
        os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
        or os.environ.get('FLASK_ENV') == 'development'
    )
    app.run(host=host, port=port, debug=debug)