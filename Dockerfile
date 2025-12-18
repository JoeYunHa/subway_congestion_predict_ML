FROM public.ecr.aws/lambda/python:3.11

# 1. 필수 라이브러리 설치 (캐시 활용을 위해 먼저 복사)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 2. 람다 함수 코드 복사
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# 3. 실행 핸들러 지정 (파일명.함수명)
CMD [ "lambda_function.lambda_handler" ]