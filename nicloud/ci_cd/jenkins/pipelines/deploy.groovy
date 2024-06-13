// File: ci_cd/jenkins/pipelines/deploy.groovy

pipeline {
    agent any

    stages {
        stage('Deploy') {
            steps {
                echo 'Deploying the project...'
            }
        }
    }
}
