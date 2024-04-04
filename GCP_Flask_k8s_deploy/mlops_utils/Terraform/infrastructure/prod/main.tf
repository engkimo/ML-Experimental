terraform {
  backend "s3" {
    bucket = "apex-test-bucket"
    key    = "apex/event_driven_job/terraform.tfstate"
    region = "ap-northeast-1"
  }
}

provider "aws" {
  region  = "ap-northeast-1"
}

variable "apex_function_datadog_logs" {}

resource "aws_iam_role" "datadog_logs_lambda_role" {
  name = "datadog_logs"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

data "aws_iam_policy_document" "datadog_logs_lambda_policy_doc" {
  statement {
    sid = "1"

    actions = [
      "s3:GetObject",
    ]

    resources = [
      "*",
    ]
  }

  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]

    resources = [
      "arn:aws:logs:*:*:*",
    ]
  }
}

resource "aws_iam_role_policy" "datadog_logs_lambda_policy" {
  name = "datadog_logs"
  role = "${aws_iam_role.datadog_logs_lambda_role.name}"

  policy = "${data.aws_iam_policy_document.datadog_logs_lambda_policy_doc.json}"
}

output "datadog_logs_lambda_role" {
  value = "${aws_iam_role.datadog_logs_lambda_role.arn}"
}

data "aws_cloudwatch_log_group" "log_group" {
  name = "datadog_logs"
}

resource "aws_cloudwatch_log_subscription_filter" "datadog_logs_filter" {
  name            = "datadog_logs_filter"
  log_group_name  = "${data.aws_cloudwatch_log_group.log_group.name}"
  filter_pattern  = "ERROR"
  destination_arn = "${var.apex_function_datadog_logs}"
}

resource "aws_lambda_permission" "datadog_logs_filter" {
  statement_id  = "datadog_logs_filter"
  action        = "lambda:InvokeFunction"
  function_name = "${var.apex_function_datadog_logs}"
  principal     = "logs.ap-northeast-1.amazonaws.com"
  source_arn    = "${data.aws_cloudwatch_log_group.log_group.arn}"
}
